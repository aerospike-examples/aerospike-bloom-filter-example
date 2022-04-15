import hashlib
import math
import struct
from typing import Callable, Iterable, Union

import aerospike
#import aerospike_helpers.batch.records # FIXME - shouldn't need this
import aerospike_helpers.batch.records as br
import aerospike_helpers.operations.bitwise_operations as bits
import aerospike_helpers.expressions as exp
import aerospike_helpers.operations.operations as operations
import aerospike_helpers.operations.expression_operations as expops


class BaseBloomHash():
    def setDesiredBytes(self, n_bytes: int):
        raise NotImplementedError

    def hash(self, values: Iterable[Union[bytearray, bytes, int, str]]):
        raise NotImplementedError


class Blake2sBloomHash(BaseBloomHash):
    def __init__(self, salt: bytes = None, desired_bytes: int = 32):
        self._salt = b"" if salt is None else salt
        self.setDesiredBytes(desired_bytes)

    def setDesiredBytes(self, n_bytes: int):
        if n_bytes < 1:
            raise ValueError("n_bytes must be > 1")

        if n_bytes <= 32:
            self._n_bytes = n_bytes
        else:
            self._n_bytes = 32

        return self._n_bytes

    def hash(self, values: Iterable[Union[bytearray, bytes, int, str]]):
        h = hashlib.blake2s(salt=self._salt, digest_size=self._n_bytes)

        for v in values:
            if isinstance(v, int):
                h.update(v.to_bytes(8, "little"))
            elif isinstance(v, str):
                h.update(bytes(v, encoding="utf8"))
            else:
                h.update(v)

        return h.digest()


def init_default_hash() -> BaseBloomHash:
    return Blake2sBloomHash()


def is_power2(value: int) -> int:
    return ((value - 1) & value) == 0


def round_up_power2(value: int) -> int:
    return 1 << int(math.ceil(math.log2(value)))


def round_down_power2(value: int) -> int:
    return 1 << int(math.floor(math.log2(value)))


class BloomSpike():
    """
    Distributed bloom filter.

    Filter is sharded across R records.
    Each record may contain S slices when hash_fn[s] sets a bit.
    """

    def __init__(self, key: tuple, capacity: int, error_rate: float,
                 max_shard_sz: int = 1024 ** 2 - 500,
                 bin_name_perfix: str = "_bf",
                 hash_init_fn: Callable[[], BaseBloomHash] = init_default_hash):
        """
        key : tuple
            how to identify the bloom filters in Aerospike
        capacity : int
            number of items expected to be stored in a filter
        error_rate : float
            false positive probability.
        max_shard_sz : int : 1024 ** 2
            size (bytes) a bloom filter may occupy on a single record
        bin_name_prefix : string : "_bf"
            record will contain several bins (on per hash_fn used) named
            "{}{}".format(bin_name_prefix, hash_fn_index)
        """
        self._key_ns, self._key_set, self._key_value = key
        self._capacity = capacity
        self._error_rate = error_rate
        self._max_shard_bit_sz = max_shard_sz * 8
        self._bin_name_prefix = bin_name_perfix
        self._bloom_hash = hash_init_fn()

        self.ans = "ans"

        max_prefix = 15 - 2

        if len(self._bin_name_prefix) > max_prefix:
            raise ValueError(
                "Prefix \"{}\" too long, max prefix size is {}".format(
                    self._bin_name_prefix, max_prefix))

        self._setOptimalSize()
        self._setOptimalNHashes()
        self._setSliceSize()
        self._setShardInfo()
        self._setDigestInfo()

    def clear(self, client: aerospike.Client):
        """
        client : connected aerospike client.
        """
        client.batch_operate(
            [self._makeKey(shard) for shard in range(self._n_shards)],
            [operations.write(self._makeSliceBin(slice_), aerospike.null())
             for slice_ in range(self._n_hashes)])

    def add(self, client: aerospike.Client, value, policy: dict=None):
        """
        client : connected aerospike client.
        value : value to add.
        policy : Aerospike operate policy
        """
        hashed = self._makeHash(value)
        offset, shard = self._readShard(hashed)
        expr = self._expNotContain(hashed, offset).compile()
        ops = []

        for i in range(self._n_hashes):
            offset, slice_bit = self._readSliceOffset(hashed, offset)
            bin_name = self._makeSliceBin(i)
            ops.append(bits.bit_resize(bin_name, self._hash_slice_sz // 8))
            ops.append(bits.bit_set(bin_name, slice_bit, 1, 1, b'\x80'))

        key = self._makeKey(shard)

        try:
            client.operate(key, ops, policy={'expressions': expr})
        except aerospike.exception.FilteredOut:
            pass

    def addAll(self, client: aerospike.Client, values:Iterable,
               policy: dict=None):
        """
        client : connected aerospike client.
        values : list of values to add.
        policy : Aerospike operate policy
        """
        shards = {}

        for value in values:
            hashed = self._makeHash(value)
            offset, shard = self._readShard(hashed)
            slices = shards[shard] = shards.get(shard, {})

            for i in range(self._n_hashes):
                offset, slice_bit = self._readSliceOffset(hashed, offset)
                bin_name = self._makeSliceBin(i)
                slice_ = slices[bin_name] = slices.get(bin_name, set())
                slice_.add(slice_bit)

        records = []

        for shard, slices in shards.items():
            ops = []
            write_record = {"key": self._makeKey(shard), "ops": ops}

            for bin_name, slice_ in slices.items():
                ops.append(bits.bit_resize(bin_name, self._hash_slice_sz // 8))
                expr = exp.BitSet(None, slice_.pop(), 1, b'\x80', bin_name)

                for slice_bit in slice_:
                    expr = exp.BitSet(None, slice_bit, 1, b'\x80', expr)

                ops.append(expops.expression_write(bin_name, expr.compile()))
            records.append(br.Write(**write_record))
        client.batch_write(br.BatchRecords(records))

    def mayContain(self, client: aerospike.Client, value, policy=None) -> bool:
        """
        client : connected aerospike client.
        value : value to check.
        policy : Aerospike operate policy
        """
        hashed = self._makeHash(value)
        offset, shard = self._readShard(hashed)
        expr = self._expMayContain(hashed, offset).compile()
        key = self._makeKey(shard)

        try:
            _, _, bins = client.operate(
                key, [expops.expression_read(self.ans, expr)])
        except (aerospike.exception.BinIncompatibleType,
                aerospike.exception.RecordNotFound):
            return False

        return bins[self.ans]

    def mayContainAny(self, client: aerospike.Client, values: Iterable,
                      policy=None) -> bool:
        """
        client : connected aerospike client.
        value : value to check.
        policy : Aerospike operate policy
        """
        batch_records = {}

        for i, value in enumerate(values):
            hashed = self._makeHash(value)
            offset, shard = self._readShard(hashed)
            key = self._makeKey(shard)
            batch_record = batch_records[key] = batch_records.get(
                key, {"key": key, "ops": []})
            expr = self._expMayContain(hashed, offset).compile()
            batch_record["ops"].append(
                expops.expression_read("{}{}".format(self.ans, i), expr))

        records = []

        for batch_record in batch_records.values():
            records.append(br.Read(**batch_record))

        res = client.batch_write(br.BatchRecords(records))
        ans = [False] * len(values)

        for batch_record in res.batch_records:
            if batch_record.result != 0:
                continue

            bins = batch_record.record[2]

            for lbl, result in bins.items():
                ix = int(lbl[len(self.ans):])
                ans[ix] = result

        return ans

    def _setOptimalSize(self) -> None:
        """
        From: https://en.wikipedia.org/wiki/Bloom_filter
              Optimal number of hash functions

              m = -((n ln e) / ((ln 2) ** 2))

              where m = self._bit_size
                    n = self._capacity
                    e = self._error_rate
        """
        error_rate = self._error_rate
        cap = self._capacity
        self._optimal_bit_sz = int(math.ceil(
            -((cap * math.log(error_rate)) / (math.log(2) ** 2))))

    def _setOptimalNHashes(self) -> None:
        """
        From: https://en.wikipedia.org/wiki/Bloom_filter
              Optimal number of hash functions

              k = (m / n) * lg(2)

              where k = self._n_hashes
                    m = self._size
                    n = self._capacity
        """
        cap = self._capacity
        bit_sz = self._optimal_bit_sz

        self._n_hashes = int(math.ceil((bit_sz / cap) * math.log(2)))

        if self._n_hashes > 100:
            raise ValueError("Too many hash functions required {}.".format(
                self._n_hashes))

    def _setSliceSize(self) -> None:
        # For performance reasons, each slice will occupy a separate bin.
        # A slice needs to be a power of 2 to ensure even distribution.

        slice_sz = self._max_shard_bit_sz // self._n_hashes
        slice_sz = ((slice_sz + 7) // 8) * 8  # round up to a byte

        if not is_power2(slice_sz):
            slice_sz = round_down_power2(slice_sz)

        self._hash_slice_sz = slice_sz

    def _setShardInfo(self) -> None:
        self._shard_sz = self._hash_slice_sz * self._n_hashes
        self._n_shards = int(math.ceil(self._optimal_bit_sz / self._shard_sz))
        self._actual_size = self._shard_sz * self._n_shards

    def _setDigestInfo(self) -> None:
        # For an even distribution, we would like for n_shards to be a power of
        # 2, but this would cause the filter to be too large. Instead, we will
        # always use a 4 byte value to determine which shard to use.
        self._dbytes_shard = 4
        self._dbytes_slice = int((math.log2(self._shard_sz) + 7) // 8)

        hash_sz_needed = self._dbytes_shard + (self._dbytes_slice *
                                               self._n_hashes)

        self._hash_sz = self._bloom_hash.setDesiredBytes(hash_sz_needed)

        self._n_times = int(math.ceil(hash_sz_needed / self._hash_sz))

    def _makeHash(self, value) -> bytes:
        return b''.join(self._bloom_hash.hash((value, bytes((i,))))
                        for i in range(self._n_times))

    def _makeKey(self, shard: int) -> tuple:
        return (self._key_ns, self._key_set, "{}{:06}".format(
            self._key_value, shard))

    def _makeSliceBin(self, slice_id: int) -> str:
        return "{}{:02}".format(self._bin_name_prefix, slice_id)

    def _readShard(self, data: bytes) -> tuple:
        dbytes_shard = self._dbytes_shard
        shard_format = ">{}s".format(dbytes_shard)
        shard_raw: bytes = struct.unpack_from(shard_format, data)[0]
        shard = int.from_bytes(shard_raw, "big",
                               signed=False) % self._n_shards
        offset = dbytes_shard

        return offset, shard

    def _readSliceOffset(self, data: bytes, offset: int) -> tuple:
        dbytes_slice = self._dbytes_slice
        slice_val_format = ">{}s".format(dbytes_slice)
        slice_raw: bytes = struct.unpack_from(slice_val_format, data,
                                              offset=offset)[0]
        slice_bit = int.from_bytes(
            slice_raw, "big", signed=False) % self._hash_slice_sz
        offset += dbytes_slice

        return offset, slice_bit

    def _expMayContain(self, hashed: bytes, offset):
        expr = []

        for i in range(self._n_hashes):
            offset, slice_bit = self._readSliceOffset(hashed, offset)
            bin_name = self._makeSliceBin(i)
            expr.append(exp.Eq(exp.BitCount(slice_bit, 1, bin_name), 1))

        return exp.And(*expr)

    def _expNotContain(self, hashed: bytes, offset):
        return exp.Not(self._expMayContain(hashed, offset))


def main():
    def ranges(start, end=None, step=None):
        for i in range(start, end, step):
            yield range(i, i + step if i + step < end else end)

    capacity = 10 ** 5
    error = 10 ** -5
    config = {"hosts": [("174.22.0.1", 3000), ("174.22.0.2", 3000)]}
    client = aerospike.client(config).connect()
    b = BloomSpike(("test", "test", "test"), capacity, error,
                   max_shard_sz=128 * 1024)
    nstep = 1000

    b.clear(client)

    assert not b.mayContain(client, b"1234")

    b.add(client, b"1234")

    assert b.mayContain(client, b"1234")

    print(b.__dict__)

    for i, v in enumerate(ranges(0, capacity, nstep)):
        print("inserting", i * nstep, "of", capacity)
        b.addAll(client, list(v))

    print("checking false negatives")

    for i, v in enumerate(ranges(0, capacity, nstep)):
        print("checking fn", i * nstep, "of", capacity)
        assert all(b.mayContainAny(client, list(v)))

    print("checking false positives")

    false_positives = 0

    for i, v in enumerate(ranges(capacity, capacity * 3, nstep)):
        print("checking fp", i * nstep, "of", capacity * 2)

        result = b.mayContainAny(client, list(v))

        if any(result):
            false_positives += result.count(True)

    print(dict(fp=false_positives, err=error))

    b.clear(client)

if __name__ == "__main__":
    import time

    now = time.time_ns()
    main()
    print("completed in {} ns".format(time.time_ns() - now))
