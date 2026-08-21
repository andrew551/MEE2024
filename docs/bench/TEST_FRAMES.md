# The curated benchmark frames

Every measurement in `docs/bench/` was produced from **82 files, 3.66 GB**,
which lived at `tests/data/fits/` until 2026-08-21 and now live at:

```
I:\MEE test frames\fits\
```

They were moved out of the repository because they are large, gitignored, and read by no test
-- the suite generates synthetic star fields in `tests/conftest.py` instead. The benchmark and
release-check commands in `docs/bench/` and `RELEASING.md` name the path above.

## Why this file exists

**The selection is not reconstructible from the source captures.** Two of the six folders are
curated subsets, and for a few hours on 2026-08-21 the only copy of *which* frames had been
deleted, which made the zwo3 ladder in `ERROR_BUDGET.md` unreproducible. Neither
`summary.json` nor the phase-bias output records its inputs. This file is the record.

| folder | taken from | how much |
|---|---|---|
| `00_23_49` | `I:\65PHQ 533MM London 2026\00_23_49` | whole folder (11 of 11) |
| `example_with_darks/070424_040415` | `I:\65PHQ 294MM Texas 2024\zenith 1\070424_040415` | whole folder (8 of 8) |
| `example_with_darks/070424_050036 darks 10s` | `I:\65PHQ 294MM Texas 2024\070424_050036 darks 10s` | whole folder (15 of 15) |
| `rasalhague` | `I:\Don Bruns TV-85 calibration` | one file of many |
| `zwo3/field` | `I:\ZWO#3 2023-10-28\Zenith-01-0.2s` | **subset**: 9 of 60 -- frames 3394-3402, the Center2 pointing less its first frame |
| `zwo3/darks` | `I:\ZWO#3 2023-10-28\Darks` | **subset**: 38 of 64 -- frames 3403-3440 |

The `zwo3` subsets are the ones that matter. `Zenith-01-0.2s` holds **six pointings of ten
frames** -- Center1, EN, ES, WN, WS, Center2 -- and every dark is `Zenith-Center2.Dark`, so a
naive `*.fit` over the source folder stacks five wrong pointings against the wrong darks. The
curated `zwo3/field` is the Center2 ten less frame 3393, and `zwo3/darks` is the first 38 of
the 64 available. A master dark from 64 frames is quieter than one from 38, so using the full
folder does not reproduce the recorded ladder.

## Verifying a copy

**Compare by content, never by name and size.** Consecutive captures from one camera write
identical filenames at identical byte counts: `00_23_49` matches three separate folders on the
source drive on name and size alone, and hashed against the neighbouring capture
(`2026-07-20\Zenith\00_25_45`), **0 of 11** files agree. The hashes below are SHA-256.

## Manifest

| file | bytes | sha256 |
|---|---|---|
| `00_23_49/Zenith_00001.CameraSettings.txt` | 1433 | `79c4b455a26c18cb42c7760f6e369729b271480efed1679f471ccfbf19eeff2c` |
| `00_23_49/Zenith_00001.fits` | 18103680 | `5e0212f04b70a061fa911169bf5a4d95a5dd8db9ea927d8bd720e819fcab6c22` |
| `00_23_49/Zenith_00002.fits` | 18103680 | `947f20abf4fd81c5cc9f10c66a66fef6a45259c0095cc92f6f45ed3de5ab261e` |
| `00_23_49/Zenith_00003.fits` | 18103680 | `b2bc09adbe4dc94c8a21b3ac1d77de2cae2417aa11d33ae0b509948aaa4d0a7e` |
| `00_23_49/Zenith_00004.fits` | 18103680 | `fc409390cffbf826c95aaff909edd8b25afec17b2d539eed23c930496433e292` |
| `00_23_49/Zenith_00005.fits` | 18103680 | `3dfd71d3bb2d3b158aeb2e5d646b96eaf776698b6b65e42dc983862c12cbdcb0` |
| `00_23_49/Zenith_00006.fits` | 18103680 | `a39e1a2ec0fda721a719fb32ce45e30075159d76bbe10ecdbf66e95a6264fc86` |
| `00_23_49/Zenith_00007.fits` | 18103680 | `c78da559291c8bb719cb1ae828124bd7bdb9f8a653f434612524379c2a93090d` |
| `00_23_49/Zenith_00008.fits` | 18103680 | `ada253d0c62376887b149cd105e1332ef7c1ebedd323dc5783a52165b77fcce2` |
| `00_23_49/Zenith_00009.fits` | 18103680 | `1940dd5862078969dc2b160751f915577d3702fbdc7dcc9b0fb1059cbd3207e0` |
| `00_23_49/Zenith_00010.fits` | 18103680 | `766f91e9d0fc39c1bbead804f14d81a9514468e421ea9d2c41fc8af86875707a` |
| `example_with_darks/070424_040415/040407_070424.txt` | 1095 | `cce805f0d5ea6c0a010df307134f37a374bf1719e023ce6c0cccf3b6e7c22546` |
| `example_with_darks/070424_040415/040407_070424_0000.fits` | 93559680 | `4e7abd80cc5da51d86f6302f5c2272a1b34ba434d69acccf615e00947c197d6e` |
| `example_with_darks/070424_040415/040407_070424_0001.fits` | 93559680 | `4d418db9a4ce7ae0cc5ffd8e7af87fb90fcd2aa7d2c593fca2ffd5cc1221c021` |
| `example_with_darks/070424_040415/040407_070424_0002.fits` | 93559680 | `dc94d27fc5cf5a7884d6d26734a253e6724038f04cba036c225340be9253472d` |
| `example_with_darks/070424_040415/040407_070424_0003.fits` | 93559680 | `68851ef0cf571c5e7f2f858b8bccf2e2317c04a0611f55f483107733e1d9cf16` |
| `example_with_darks/070424_040415/040407_070424_0004.fits` | 93559680 | `9e50764f0c2e75ab4175173434d70ec4b7dd7d25f54acb9891fe3058acfe90bc` |
| `example_with_darks/070424_040415/040407_070424_0005.fits` | 93559680 | `cdeadda9b8beb704f16e1a74f2f225f1affdacad6f0e6e0b3004daf06be6ad48` |
| `example_with_darks/070424_040415/040407_070424_0006.fits` | 93559680 | `a862090b111ac3381bc88da699c84825dfbe2ff89783f42c0e219f7a857c60e9` |
| `example_with_darks/070424_050036 darks 10s/044755_070424.txt` | 1099 | `b0a6fdb4c0002cf4ac2ffaa50e81d65cf5be644e1d7e6db3751f5b512f5400cc` |
| `example_with_darks/070424_050036 darks 10s/044755_070424_0000.fits` | 93559680 | `95473676830ffd04a78e8b8ad305fbae708c4467dfbe50a16c9981b39fd04f94` |
| `example_with_darks/070424_050036 darks 10s/044755_070424_0001.fits` | 93559680 | `25aee3a8ef6627a58c972d68cd734dc32f513de43a5aba7ec97f3dd989a464cb` |
| `example_with_darks/070424_050036 darks 10s/044755_070424_0002.fits` | 93559680 | `44eed532ad4deb07cf5ace36fc20a1e64c14fd8ffc87d6f796d0d0a4ba5695a4` |
| `example_with_darks/070424_050036 darks 10s/044755_070424_0003.fits` | 93559680 | `3bd22d5be125fa39bf9da5b109c4e8871d275f7da79ee24c528cf3ea59d0df6a` |
| `example_with_darks/070424_050036 darks 10s/044755_070424_0004.fits` | 93559680 | `19b8e752fe663dc3a0fc1260f6dafdac470e27e633febc21b4b4e1dc9d71eab1` |
| `example_with_darks/070424_050036 darks 10s/044755_070424_0005.fits` | 93559680 | `cceb5e2dd23f10d7199abce5e08ec67b5e564ee48a93edfec8b67a30c0a96f23` |
| `example_with_darks/070424_050036 darks 10s/044755_070424_0006.fits` | 93559680 | `8c2a79b8c0b78af844aff9e9e89dc18d3d51b28d839235fb746f3972912ab25c` |
| `example_with_darks/070424_050036 darks 10s/050036_070424.txt` | 1068 | `dab4332a1558c1b1993c8fe2a03e126d9a768a2cdc749610b34fd2a33ae3f1db` |
| `example_with_darks/070424_050036 darks 10s/050036_070424_0000.fits` | 93559680 | `9e21be21bd7c4e0f732d3760232b80e6c9f95511549b9d44bb832538fe603b22` |
| `example_with_darks/070424_050036 darks 10s/050036_070424_0001.fits` | 93559680 | `fc44d668e4035e173992a08a529d106c04fa61fe42cf961efb01e980961c64f6` |
| `example_with_darks/070424_050036 darks 10s/050036_070424_0002.fits` | 93559680 | `5bf15d5285d0a291410af93bf5edce5cdefc45639a1bc490359b7ef651c33019` |
| `example_with_darks/070424_050036 darks 10s/050036_070424_0003.fits` | 93559680 | `b431544d2c6db6d801d8fe77786a2e2c3efef6fa15a245b3e766d117bbe2a8c4` |
| `example_with_darks/070424_050036 darks 10s/050036_070424_0004.fits` | 93559680 | `563a8eea8f5cb2cea29cd2ddb0e2d37b5e48aa43dd48dfe17ba3173b6219f9c5` |
| `example_with_darks/070424_050036 darks 10s/050036_070424_0005.fits` | 93559680 | `2647a20639652b338991d7af38a00f98e195ed48e537d4209c45b2dc06b6a618` |
| `rasalhague/Rasalhaguemean50.fit` | 65563200 | `a364a57919e9678f2d244eb0e2f88293262ec92cc43dfa5cbe01ab32e784fd2f` |
| `zwo3/darks/MEE2024.00003403.Zenith-Center2.Dark.fit` | 32785920 | `98781028d2a2c64e652140e9c0f945951318a091a84b8344c8131996a8036b84` |
| `zwo3/darks/MEE2024.00003404.Zenith-Center2.Dark.fit` | 32785920 | `4d5a159888ec730f81f99678d9bb20af52226151c64b31242e87412e430360f6` |
| `zwo3/darks/MEE2024.00003405.Zenith-Center2.Dark.fit` | 32785920 | `89d6bbfe967afeec7ceb895d7cf21de46496ad6103b778519733573ce2a01617` |
| `zwo3/darks/MEE2024.00003406.Zenith-Center2.Dark.fit` | 32785920 | `485cf562f4d99aba90b66b9d067ec973fb7df9b344f28f5e28569cc7a33b0fcc` |
| `zwo3/darks/MEE2024.00003407.Zenith-Center2.Dark.fit` | 32785920 | `bfb52ef3e40f98322dcce12069fa37964b1764693ac70b35d59c0892b126e0ca` |
| `zwo3/darks/MEE2024.00003408.Zenith-Center2.Dark.fit` | 32785920 | `56af655d4ee1b6e983f23fd6f9efb1093ff2a816ee8e3c23d86eac2ddc0f7dde` |
| `zwo3/darks/MEE2024.00003409.Zenith-Center2.Dark.fit` | 32785920 | `2cfafaa097668e6e1cedab28e5a50cb73aa81250390f9206f4f71fe3e7b556fe` |
| `zwo3/darks/MEE2024.00003410.Zenith-Center2.Dark.fit` | 32785920 | `894acb212d12a559390524c414a8000db0eeb3ef340f34943eb7a311d9cf574e` |
| `zwo3/darks/MEE2024.00003411.Zenith-Center2.Dark.fit` | 32785920 | `ec5c07c0e2af220987fcacb0f285130d0743bd29582575a8ed5799fae9c73cd9` |
| `zwo3/darks/MEE2024.00003412.Zenith-Center2.Dark.fit` | 32785920 | `073c64149d177cf447d550068832f12dec21ec69c59026d58bf9a6fc6a3db1ac` |
| `zwo3/darks/MEE2024.00003413.Zenith-Center2.Dark.fit` | 32785920 | `fd03ea34813436935d448084fbb61246da0eaa4977449c7611faec664e375ec0` |
| `zwo3/darks/MEE2024.00003414.Zenith-Center2.Dark.fit` | 32785920 | `54bb2f9123b13e0a14b5b387c0dc6eca6457a38165b0a2bda9c9dacf8aeb4824` |
| `zwo3/darks/MEE2024.00003415.Zenith-Center2.Dark.fit` | 32785920 | `f1537a3ccc7e272fa80b566bb511f91ade82d77beef7885ccb73054ef6f592ee` |
| `zwo3/darks/MEE2024.00003416.Zenith-Center2.Dark.fit` | 32785920 | `e798ca5eac69f260689f76987566a18d4ea6ba3f1d31680010d0f94a95d06f23` |
| `zwo3/darks/MEE2024.00003417.Zenith-Center2.Dark.fit` | 32785920 | `f8801e21086ec9132027104b48d38c02bfeab4bec8186979441d54f64c79b56d` |
| `zwo3/darks/MEE2024.00003418.Zenith-Center2.Dark.fit` | 32785920 | `ecffa0117676114b2ffc562db25461fd8436bbe066dae1d7a0361bcf1ebe265a` |
| `zwo3/darks/MEE2024.00003419.Zenith-Center2.Dark.fit` | 32785920 | `93dbad6d35df452e1aa33808112a5bf28d2addc0fcb20acf80a34dc75383f1a5` |
| `zwo3/darks/MEE2024.00003420.Zenith-Center2.Dark.fit` | 32785920 | `15955c2adb9108a3066f732971cb2f045998733062d9fcb4f048f383b10ae581` |
| `zwo3/darks/MEE2024.00003421.Zenith-Center2.Dark.fit` | 32785920 | `ab17ed2529bdaa75fe34eb9fe3d491687c00c3014b8821f5df0bfb2b8406223e` |
| `zwo3/darks/MEE2024.00003422.Zenith-Center2.Dark.fit` | 32785920 | `242066082a0fc5a169ee5fcea3c40c353b0e79d55902f4ca3ecea92a16637140` |
| `zwo3/darks/MEE2024.00003423.Zenith-Center2.Dark.fit` | 32785920 | `4dc2470755b6420ed78a144fd8ce746f9bf5b338d891d1c09b886543efe51dab` |
| `zwo3/darks/MEE2024.00003424.Zenith-Center2.Dark.fit` | 32785920 | `d7e40a464cdb246422464a7316c15400e0787a04e78b8ab06935ba187caf5e7b` |
| `zwo3/darks/MEE2024.00003425.Zenith-Center2.Dark.fit` | 32785920 | `30caa0b9facf37729eb671fc22fc8281f958372ba386e717ba822d0828812997` |
| `zwo3/darks/MEE2024.00003426.Zenith-Center2.Dark.fit` | 32785920 | `1b00581d9dc4230c1da89cfe3539eeb736382b233b1750634705fdf0ff41fa32` |
| `zwo3/darks/MEE2024.00003427.Zenith-Center2.Dark.fit` | 32785920 | `404a51b84f990f55f803b709aecb63e3eb3b82f338bd84c08bc6a6b21035ca30` |
| `zwo3/darks/MEE2024.00003428.Zenith-Center2.Dark.fit` | 32785920 | `a2528527cf71cfa4781dfe655f0a2f26da6bc29fa336ebbb7e30b44610153da3` |
| `zwo3/darks/MEE2024.00003429.Zenith-Center2.Dark.fit` | 32785920 | `282f0298bf5dcc8b99a0a4fc974f9606ccdfdf0ab401cbd688b3b56bda882f3f` |
| `zwo3/darks/MEE2024.00003430.Zenith-Center2.Dark.fit` | 32785920 | `775085cbef99995a3e78b713e5417b171227c5a437152e953d8301068366e200` |
| `zwo3/darks/MEE2024.00003431.Zenith-Center2.Dark.fit` | 32785920 | `99372d5368fae3099d22f2e59c3eca9cb93ac587dd877cd8dd984fa1e3806cdc` |
| `zwo3/darks/MEE2024.00003432.Zenith-Center2.Dark.fit` | 32785920 | `ab7cde2a37f393c4cd51b3c8783fbd880d05da555137cc4e3f309ea6f2b1a4d8` |
| `zwo3/darks/MEE2024.00003433.Zenith-Center2.Dark.fit` | 32785920 | `8ffa5bf8b434719897ca4fc0d8329c57b8c257c0f1ebaaa37f8ebd0af5b4b1a4` |
| `zwo3/darks/MEE2024.00003434.Zenith-Center2.Dark.fit` | 32785920 | `581bb9f7650ee70f3778adfcb97916293f7cfe45ff76d4743ac1521397af7999` |
| `zwo3/darks/MEE2024.00003435.Zenith-Center2.Dark.fit` | 32785920 | `0c15d8b431804cd2dddae33ad31646fa5873ee28540089dad606e9f7f609b951` |
| `zwo3/darks/MEE2024.00003436.Zenith-Center2.Dark.fit` | 32785920 | `b7a4aca7813d36d1fecc93aa8caf7f86fc190ffc38cb95b0f3a396eb456a36fb` |
| `zwo3/darks/MEE2024.00003437.Zenith-Center2.Dark.fit` | 32785920 | `461135ff2c9982af3996f14bafe11978d8f83dd7044a82f0792f6c5b1ae2df36` |
| `zwo3/darks/MEE2024.00003438.Zenith-Center2.Dark.fit` | 32785920 | `44418e256d0153b6961a6f944728ff505e17f25b2afc9df086d3905e3e0e3c4d` |
| `zwo3/darks/MEE2024.00003439.Zenith-Center2.Dark.fit` | 32785920 | `8c2147661c6f01d53b7f86f76656428ca134fbc2e798d02fbf4d3f68982caea6` |
| `zwo3/darks/MEE2024.00003440.Zenith-Center2.Dark.fit` | 32785920 | `7171d1d87238e95886ebe71fd5f9a0f7f19b76f76f977efcc3ec596724601e3a` |
| `zwo3/field/MEE2024.00003394.Zenith-Center2.fit` | 32785920 | `bc0c3ab768208db2f051625a3582566e020c3c315efb97afca20ffada7ac6a57` |
| `zwo3/field/MEE2024.00003395.Zenith-Center2.fit` | 32785920 | `c68f54af2f18f4f86c28765fe1916ce26c41e24bb3439bb7bb8f68db26f0d125` |
| `zwo3/field/MEE2024.00003396.Zenith-Center2.fit` | 32785920 | `878771e4ad7ea2b5422374d32651f759560588bf1ba4ea03f256b045f1e9cd7c` |
| `zwo3/field/MEE2024.00003397.Zenith-Center2.fit` | 32785920 | `340b516bf97315a63c9120da58384a04614d79991c4193e626a7bc6bb235c118` |
| `zwo3/field/MEE2024.00003398.Zenith-Center2.fit` | 32785920 | `1f8d5167890ce503d6a364ce15d6802b00ec2a12921226b0ec060b23ce860c61` |
| `zwo3/field/MEE2024.00003399.Zenith-Center2.fit` | 32785920 | `ac6458f580b4c2016840d8ddc6e7992d147b58c2b248eb2646939dbc2e831dc8` |
| `zwo3/field/MEE2024.00003400.Zenith-Center2.fit` | 32785920 | `d086d55e019d277471af6cdee62002a19f5749b2d7d61d1dbd172f0e6687cfbd` |
| `zwo3/field/MEE2024.00003401.Zenith-Center2.fit` | 32785920 | `02b50d9977aafdacfe4861638be61790eec0c6933529e76dab4b254a411da7ab` |
| `zwo3/field/MEE2024.00003402.Zenith-Center2.fit` | 32785920 | `778bea83bdbcb5104d8a4b7e2af4ef9ea5bb15865deab1471c740a114f4bcafd` |
