## [0.4.1] - 2026-08-25

### Fixes

- Report the probe's own error instead of an allocation failure ([86d0d65](https://github.com/walcark/xsweep/commit/86d0d655a256ef14f71566ef9269c035124de50e))
## [0.4.0] - 2026-08-24

### Build

- Scaffold the sphinx-gallery example site ([e489cf1](https://github.com/walcark/xsweep/commit/e489cf1f6d09a818d345afb99958ab90b7b96989))
- Add matplotlib to the dev pixi feature ([11c76a6](https://github.com/walcark/xsweep/commit/11c76a63e7c87500960e79d65b875058eeddd4cf))
- Rebuild the docs site on shibuya, with the guide and reference on it ([de8a2b6](https://github.com/walcark/xsweep/commit/de8a2b6ff35d55e9ea93ba0f2623e7bc718546f3))

### CI

- Deploy the example gallery to GitHub Pages, link it from the docs ([700cd3e](https://github.com/walcark/xsweep/commit/700cd3e58fb5f220ac717210657913c782f2eff9))
- Release to pypi on tag push, guarded by the version and the gate ([3181afc](https://github.com/walcark/xsweep/commit/3181afcea280c0181acb3fd10d4c235dd3504130))

### Chores

- Release 0.4.0 ([bec4254](https://github.com/walcark/xsweep/commit/bec42543c5235c33b08a88a099bd4f7ddb7cf71c))

### Documentation

- Rebuild the gallery around an engine that cannot be vectorised ([294b1a4](https://github.com/walcark/xsweep/commit/294b1a4cee354879d37aa4fcc95e6557765f4d76))
- Add gallery page 02, the store as cache and resume path ([947528b](https://github.com/walcark/xsweep/commit/947528b6533346de80e1ca92249a7609be6ad291))
- Add gallery page 03, zip against product ([1b95e8a](https://github.com/walcark/xsweep/commit/1b95e8aa9df804648a6b483fa79e10a35ce67682))
- Add gallery page 04, deduplication on a scene ([ea3e413](https://github.com/walcark/xsweep/commit/ea3e413e4bf6ed0c25f2bb2740faeb6059995c3c))
- Add gallery page 05, handing over a whole spectrum with vec ([769cbad](https://github.com/walcark/xsweep/commit/769cbadfdd6b63530a0ddff7f8343a7f2b918ff2))
- Add gallery page 06, const and the axis that must not be cut ([da1eeea](https://github.com/walcark/xsweep/commit/da1eeeaae52dbc5fcfbe6d961d4018fc99b94d6a))
- Add gallery page 07, policy changes cost and never the result ([56a6c9d](https://github.com/walcark/xsweep/commit/56a6c9d7618392e23524ed5fdc36b59b3ae6d1ef))
- Add gallery page 08, SweepModule and state built once ([5a8bc0b](https://github.com/walcark/xsweep/commit/5a8bc0b63fecc05cc944e39da415cedad6c34ca0))
- Add gallery page 09, recording, retrying and resuming failures ([844b0d3](https://github.com/walcark/xsweep/commit/844b0d33fcee955ab366e4a7ca6212b56f5902d2))
- Add gallery page 10, replication through a seed variable ([b21c19c](https://github.com/walcark/xsweep/commit/b21c19c94adb57b6be164408bf95828560a90f7c))
- Rework the guide into site pages, reframed on the case xsweep is for ([8cb5f61](https://github.com/walcark/xsweep/commit/8cb5f6166831cfa0a591d5b92500516d8c6decfe))
- Cut the README down to a pointer at the site ([1cc1009](https://github.com/walcark/xsweep/commit/1cc1009aa0370d717649642b8794fd9f38b9e065))

### Features

- Add benchmarks harness plumbing ([fd7d921](https://github.com/walcark/xsweep/commit/fd7d921c2d67ff69aa3801f2f741bdc0d94c9b1e))
- Add benchmark case 1, Beer-Lambert transmission ([bc5ee47](https://github.com/walcark/xsweep/commit/bc5ee47e9509c8857e482b5bdc2df2318345fc8b))
- Add benchmark case 2, Rayleigh optical depth spectrum ([397f8bd](https://github.com/walcark/xsweep/commit/397f8bd049fb429bf18265c9a57ea58201a28e8c))
- Add benchmark case 3, dedup on a pixel map with a store ([edba768](https://github.com/walcark/xsweep/commit/edba7689376a68d916ad0b84927106040bca30de))
- Add benchmark case 4, band integration against synthetic SRFs ([1415345](https://github.com/walcark/xsweep/commit/1415345a5273ce112ab8f380cad82c0d3507d4c9))
- Add benchmark case 5, TOA reflectance over a big grid ([3bc38ee](https://github.com/walcark/xsweep/commit/3bc38ee21990030a70b49ced9f4863f3d233da2a))
- Add benchmark case 6, Henyey-Greenstein phase function ([16fc2f1](https://github.com/walcark/xsweep/commit/16fc2f1c46790d0ded80e58447802bb48bd5f954))
- Add benchmark case 7, hyperspectral cube dedup at scale ([c318bc3](https://github.com/walcark/xsweep/commit/c318bc315384c34f5be51b94c4135780ff4f888b))
- Add benchmark case 8, Monte-Carlo ensemble over seed ([36545df](https://github.com/walcark/xsweep/commit/36545df63790c35dcee513462bd4d5da28632d8e))
- Add benchmark case 9, LUT + SweepModule + process executor ([35b8bc5](https://github.com/walcark/xsweep/commit/35b8bc55808d4503ac0d292d6fd76b15905c4dfd))
- Add benchmark case 10, resume after interruption ([805773d](https://github.com/walcark/xsweep/commit/805773d22125fe45f54090e56cbbb3c6b5df20f1))
- Add the benchmark evolution page and refresh recorded results ([133dd6b](https://github.com/walcark/xsweep/commit/133dd6be513630f62287a39b3039f976910521d5))
- Add the radiative-transfer solvers the gallery will run on ([9eb4eb0](https://github.com/walcark/xsweep/commit/9eb4eb0112fdbc82a218ba499c47a58585a1c02a))
- Deliver loop variables a group at a time with the batch clause ([2ec741f](https://github.com/walcark/xsweep/commit/2ec741fea70e68b77379928f52266a7ff77b0693))

### Fixes

- Anchor the gallery ignore pattern to the start of the file name ([64b1b8b](https://github.com/walcark/xsweep/commit/64b1b8b9258ce19a3f50e74412ff4647b4b5f35c))

### Refactoring

- Split the frozen benchmark cases from the example gallery ([474ddf0](https://github.com/walcark/xsweep/commit/474ddf08b5eb6d7d0fb48df4aed622d402654fc1))
## [0.3.0] - 2026-07-30

### Chores

- Add classifiers, keywords and project URLs ([2a4b4b0](https://github.com/walcark/xsweep/commit/2a4b4b08eeadc80aa5c18480ca4791bb7643795a))
- Bump version to 0.3.0 ([e8701eb](https://github.com/walcark/xsweep/commit/e8701ebfd74acfdc8db28d90bd546b103fbc8e9b))

### Documentation

- Regenerate CHANGELOG.md ([3036027](https://github.com/walcark/xsweep/commit/3036027921ffd7381394e3d760c10d1102c8244d))
- Regenerate CHANGELOG.md for v0.3.0 ([6f9eabd](https://github.com/walcark/xsweep/commit/6f9eabdd9cf48b7e8842363f6f54e5833c89bd44))

### Features

- Let chunks auto-size a named vec batch from a memory budget ([dd15676](https://github.com/walcark/xsweep/commit/dd1567613f8da130b49813a539130b1db030d449))

### Tests

- Cover executor selection and dispatch failures ([4d46b9a](https://github.com/walcark/xsweep/commit/4d46b9ac582687e784d0d302484610a6fe4ab466))
- Cover the lock's edge cases (100%) ([e48eada](https://github.com/walcark/xsweep/commit/e48eada99484e8b66f2b280a135f9317ddd3aa11))
## [0.2.0] - 2026-07-30

### CI

- Run the full pixi gate on push and pull request ([8cb0118](https://github.com/walcark/xsweep/commit/8cb011801eee4a4e41f3824eebd43f15934b65b3))
- Wire up Codecov and its README badge ([bd9da6f](https://github.com/walcark/xsweep/commit/bd9da6f6fbd4166807ae5531afa70462c2ce04dc))

### Chores

- Add project tooling and test fixtures ([ec026bd](https://github.com/walcark/xsweep/commit/ec026bdaba7c42ed53fa76a0359c41a4506b30ed))
- Init spec-kit scaffolding ([954b8a1](https://github.com/walcark/xsweep/commit/954b8a1f8c4be04ab12b1f41fccea2bc56981b09))
- Relicense under Apache-2.0 ([25a5b0a](https://github.com/walcark/xsweep/commit/25a5b0a1b3681c633ef64c550dcce037cfea462b))
- Generate CHANGELOG.md with git-cliff ([48d746f](https://github.com/walcark/xsweep/commit/48d746f5d05ea32f070135922e38845a5a708026))
- Stop ignoring pixi.lock ([fac899a](https://github.com/walcark/xsweep/commit/fac899a201230855d3e46608b4fe145a6682b6ff))
- Correct the author's name to Kévin Walcarius ([863ecd8](https://github.com/walcark/xsweep/commit/863ecd8f2d1e80fb036bdac9cb16f07e882ac08a))
- Bump version to 0.2.0 ([7ee9df9](https://github.com/walcark/xsweep/commit/7ee9df913804f0e847747c9a4238df16fae7e76c))

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
- Add badges and a License section to the README ([161a9ae](https://github.com/walcark/xsweep/commit/161a9aeb2d2cc3e62e2d44fabf81f781b2c3e312))
- Regenerate CHANGELOG.md for v0.2.0 ([0ee1920](https://github.com/walcark/xsweep/commit/0ee19200153164fc4c8ded295528685deeaeb48f))

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
