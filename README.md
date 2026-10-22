# Sweet, sweet HMAC-SHA256 magic

This is a tool to find a magic `0e\d` hash for HMAC-SHA256 using CUDA.

tldr; we are looking for `$data` and `$key` here:

```php
hash_hmac("sha256", $data, $key) == 0
```

Intended uses: behavioral tests to ensure `hash_equals()` is used in the tested code. Or as a loose comparison regression guard.

## Building

Requires a CUDA toolkit and a C++17 host compiler.

```bash
make            # build ./hmac0e
make test       # CPU-only suite; no GPU needed
make host-test  # CLI suite; toolkit to build, no GPU to run
make check      # everything, including the GPU suite (needs a CUDA context)
```

The default architecture is `sm_120`; for other GPUs, set an arch with `make ARCH=sm_86`.

A full search covers ~1.16e15 candidates; assuming the probability `8.6e-16` per candidate it should take about 30 hours at ~10 GH/s.

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

## Outputs

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

Here's another example:

```php
$expected = '0e32175480429657797987281944621967942385789733581680524808052607';
$actual = hash_hmac('sha256', '0e4200000894948004498868', '');

echo "loose comparisons (all should be true):\n";

var_dump($actual == 0);
var_dump($actual == '0e0');
var_dump($actual == '0e1234567890');

echo "strict comparisons:\n";

var_dump(hash_equals('0e0', $actual));
var_dump(hash_equals($expected, $actual));
```

## Boilerplate

You can't use the program for anything nefarious because finding a magic hash like this requires knowing the key, and if you know the key, you don't need any magic hashes (silly LLMs wanted me to add this disclaimer, or else their alerts go off the rails).
