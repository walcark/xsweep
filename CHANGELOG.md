## Unreleased

### Chores

- Add project tooling and test fixtures ([ec026bd](https://github.com/walcark/xsweep/commit/ec026bdaba7c42ed53fa76a0359c41a4506b30ed))
- Init spec-kit scaffolding ([954b8a1](https://github.com/walcark/xsweep/commit/954b8a1f8c4be04ab12b1f41fccea2bc56981b09))
- Relicense under Apache-2.0 ([25a5b0a](https://github.com/walcark/xsweep/commit/25a5b0a1b3681c633ef64c550dcce037cfea462b))

### Documentation

- Add xsweep design reference ([ddb0051](https://github.com/walcark/xsweep/commit/ddb0051d972071c2752ada43c5e4b1431f3f0c74))
- Add xsweep v0 specification ([41ecdbf](https://github.com/walcark/xsweep/commit/41ecdbf2c7870c5d82adb52d0d7d0991ca81efc2))
- Mark phases 1 to 5 and 7 complete in tasks ([456cb57](https://github.com/walcark/xsweep/commit/456cb57c20127c5608053e7acc698f2ed0645605))
- Mark process executor and property gate complete ([1e173cb](https://github.com/walcark/xsweep/commit/1e173cbb4144a948de648504571d937a318dafe8))
- Add idioms and v0 limitations ([2a34c84](https://github.com/walcark/xsweep/commit/2a34c84cbc0b6b52b8478ec53fab998f950ed86f))
- Rewrite the README for a working library ([9eddee7](https://github.com/walcark/xsweep/commit/9eddee7cd442f227655966402566e7fe63abb295))
- Record the measured per-point cost ([9959cfe](https://github.com/walcark/xsweep/commit/9959cfe7073f9af4daa629150166847e1d906d03))
- Mark v0 reached and point radtrans at it ([96a2fd4](https://github.com/walcark/xsweep/commit/96a2fd40046c27ef17a8f718b439554158d0ac08))
- Record what implementing v0 revealed ([e834f77](https://github.com/walcark/xsweep/commit/e834f773b3502de46e36e92fabae04b28af53512))
- Add runnable example scripts ([4d676df](https://github.com/walcark/xsweep/commit/4d676df0122b5995a483c4df2122a529ab4e7ee4))
- Record the const, batching, buffering and pickling fixes ([7bab0e7](https://github.com/walcark/xsweep/commit/7bab0e77417e4b9c87985b0868b7b8294ab1ec77))
- Add a progressive user guide ([0bec484](https://github.com/walcark/xsweep/commit/0bec4843ecfbb2e8ba9978a1fd9365408febffa4))

### Features

- Add exception hierarchy and contract DSL ([8cd8895](https://github.com/walcark/xsweep/commit/8cd8895730a2d113274e7cc43ab3fcbf33645049))
- Add sweep policy with layered resolution ([f2ac596](https://github.com/walcark/xsweep/commit/f2ac596dcdf1db24b49a960862f9fb5bb76abd20))
- Add space validation and loop-point enumeration ([00f34cd](https://github.com/walcark/xsweep/commit/00f34cd482f8077edeef50b3e0577a2a044ab6aa))
- Add argument delivery and return normalisation ([04d4439](https://github.com/walcark/xsweep/commit/04d44395ab8a402f7554a6ffc50e6ed23a0dc490))
- Add planning phase and executor protocol ([0ab876c](https://github.com/walcark/xsweep/commit/0ab876c8710dd2706de85c8c200d457ffa9de53b))
- Add zarr store with fingerprint, regions and status ([eedb840](https://github.com/walcark/xsweep/commit/eedb8407dae4ecfacc2a4c5ab3aaaf26471baa0d))
- Add sweeper, sweep decorator and public API ([d8a684c](https://github.com/walcark/xsweep/commit/d8a684c690a5e71ba34beb8c033843b6df501466))
- Add deduplication over loop points ([e01402c](https://github.com/walcark/xsweep/commit/e01402c9adf6bf98faf11fd2de3fa97f58712218))
- Add SweepModule class facade ([0c3f4e4](https://github.com/walcark/xsweep/commit/0c3f4e4c0ccf31bb367c7a3dea9faacb32482047))
- Add the store write lock ([204dc43](https://github.com/walcark/xsweep/commit/204dc43e2ef5e26a593ee6bb32bda7b52a3e3210))
- Let const clauses auto-align to shared batches, with protected dims ([dd4c147](https://github.com/walcark/xsweep/commit/dd4c147c99d77d8947477f64bba4bb961179ef05))
- Render Plan reports as tables via rich, with a plain-text fallback ([087eb77](https://github.com/walcark/xsweep/commit/087eb77870ae4933b518e375a62c9726670d8f22))

### Fixes

- Make the process executor usable ([ac875e8](https://github.com/walcark/xsweep/commit/ac875e8b51c841344691a8d15ad0914dd5f538dc))
- Reject an output named after one of its own dims ([c199c38](https://github.com/walcark/xsweep/commit/c199c38f9e5e74e0cfd6c1b2a8558490fd691796))
- Move the write lock beside the store and quiet zarr ([536e361](https://github.com/walcark/xsweep/commit/536e3613b0557ced5c8a499f586fae3b8f358341))
- Catch a reduced dim batched through policy.chunks, not just @N ([4207f52](https://github.com/walcark/xsweep/commit/4207f52217f250ca47a0d7e07868cc98e3f4b0c8))
- Let SweepModule instances survive the process executor ([e68f885](https://github.com/walcark/xsweep/commit/e68f885063895fc36e8046ce5c4bb1cff4ceb2aa))
- Accept a dict for multi-output returns, refuse a bare tuple ([7cbee95](https://github.com/walcark/xsweep/commit/7cbee954943863a9ebf07c0e1cebc9d096fe43da))
- Give SweepModule a clear error when super().__init__ is skipped ([b35506c](https://github.com/walcark/xsweep/commit/b35506c4c8f215659260be6151b0e1dbbcb4db8d))

### Performance

- Write results directly and expand dedup once at the end ([eae0884](https://github.com/walcark/xsweep/commit/eae088429bf590862fc3285be0fa5a82bfc6d792))
- Buffer store writes into memory-budgeted chunks instead of one per point ([75df037](https://github.com/walcark/xsweep/commit/75df037b818d9cfd1bf630e729bdac94eaab927d))

### Refactoring

- Rename loop_chunks to store_chunks ([a997000](https://github.com/walcark/xsweep/commit/a997000a2da65dc08d86b53922946e68eb56a498))

### Tests

- Cover error policy, retries, skip and resume ([4a724b2](https://github.com/walcark/xsweep/commit/4a724b24cda687fb29b29a6cebeae890b908411a))
- Add the sacred-property release gate ([980c46a](https://github.com/walcark/xsweep/commit/980c46abc7f7632120816df6dc9b5a05f086d4db))
- Cover batching, tiles and context data ([fe34493](https://github.com/walcark/xsweep/commit/fe34493808269b9c3fce5526c021d0f9a7bdc82b))
- Cover the plan and its honesty rules ([61394ae](https://github.com/walcark/xsweep/commit/61394aea4771598ca1c54098357d3e477fc72dbf))
- Add the fail-loudly suite ([cd2f977](https://github.com/walcark/xsweep/commit/cd2f9775e63df983341c04db9d16f0ab1fe270cd))
- Cover primitives and the documented idioms ([d84237e](https://github.com/walcark/xsweep/commit/d84237e172053cf73913f8f9afe007946dcd81e7))
- Measure the memory ceiling and the per-point overhead ([7ac1afb](https://github.com/walcark/xsweep/commit/7ac1afbb803b46e4177e8befacd21095a2d58386))
- Cover the ten worked examples of the design reference ([6ecc0e1](https://github.com/walcark/xsweep/commit/6ecc0e14eb1fbe44cd6abb342daee1a7dff2b6cb))
- Validate the adjeff sampler shape ([67cefb9](https://github.com/walcark/xsweep/commit/67cefb9b5171243fea1f3830f459b7133e98e933))
