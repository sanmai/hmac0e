# Sweet, sweet HMAC-SHA256 magic

This is a tool to find a magic `0e\d` hash for HMAC-SHA256 using CUDA.

tldr; we are looking for `$a` and `$b` here:

```php
hash_hmac("sha256", $a, $b) == 0
```

Intended uses: behavioral tests to ensure `hash_equals()` or `===` are used in the tested code.

It can't be used for anything nefarious because finding a magic hash like this requires knowing the key, and if you know the key, you don't need any magic hashes.

## Building

Requires a CUDA toolkit and a C++17 host compiler.

```bash
make            # build ./hmac0e
make test       # CPU-only suite; no GPU needed
make host-test  # CLI suite; toolkit to build, no GPU to run
make check      # everything, including the GPU suite (needs a CUDA context)
```

The default architecture is `sm_120`; for other GPUs, set an arch with `make ARCH=sm_86`.

The prefix length must satisfy `((len + 12) % 64) <= 47`. The search space is the full 2^64 counter range.

A `--digits 62` search covers ~1.16e15 candidates in expectation (probability `8.6e-16` per candidate) - about 30 hours at ~10 GH/s.

## Running

```bash
./hmac0e --key hmac-sha256-magic-hash-key --prefix 'magic-hash-' --checkpoint checkpoints/run2.txt --hits-file hits.txt
```

It automatically resumes the known checkpoint. To require an existing checkpoint (i.e., to abort if a checkpoint is missing), add `--resume`.

It will output something like this:

```
$ ./hmac0e --key hmac-sha256-magic-hash-key --prefix 'magic-hash-' --checkpoint checkpoints/run2.txt --hits-file hits.txt
device 0: NVIDIA GeForce RTX 5090 (170 SMs)
key 26 bytes, prefix 11 bytes, digits 62, body = prefix || 20 digits
grid 1020 x 256, 512 counters/thread -> 1.337e+08 per launch
 44582.7s    9.75 GH/s  4.3486e+14 candidates  0 hits
HIT counter=434871031216900 body=magic-hash-00000434871031216900 body_hex=6d616769632d686173682d3030303030343334383731303331323136393030 digest=0e54716623902408569054496933889872294553517404006854343223678539

done: 434871100000000 candidates in 44583.60s (9.75 GH/s), 1 hit
```

And then you can use it as such:

```php
$expected = '0e54716623902408569054496933889872294553517404006854343223678539';
$actual = hash_hmac('sha256', 'magic-hash-00000434871031216900', 'hmac-sha256-magic-hash-key');

echo "loose comparisons (all should be true):\n";

var_dump($actual == 0);
var_dump($actual == '0e0');
var_dump($actual == '0e1234567890');

echo "strict comparisons:\n";

var_dump(hash_equals('0e0', $actual));
var_dump(hash_equals($expected, $actual));
```

That outputs:
```
loose comparisons (all should be true):
bool(true)
bool(true)
bool(true)
strict comparisons:
bool(false)
bool(true)
```
