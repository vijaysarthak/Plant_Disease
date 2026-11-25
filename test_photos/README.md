# Test photos for the Plant Disease Advisor app

None of these photos was used to TRAIN the model or to set its thresholds:
* lab photos  = PlantVillage TEST split (leaf-grouped, seed 42 - the same split as your Colab run)
* field photos = PlantDoc photos NOT picked for field calibration (25 per group, seed 42)
* rice photos = a crop the model has never seen
* not-a-leaf  = images made by a script (grey, noise, sky colour)

| Folder | What it tests | Answer of the DEMO model |
|---|---|---|
| 1_lab_photos | 2 lab photos per group (G0-G5) | advise, correct group (12/12) |
| 2_field_advise_correct | field photos where advice is given and right | advise, correct |
| 3_field_refer_to_expert | field photos where the groups can't be narrowed down | refer to KVK |
| 4_field_unknown | field photos that look unfamiliar (energy too high) | photo not recognised |
| 5_field_advise_WRONG | field photos where advice is given but WRONG (the ~20% IAR) | advise, wrong group |
| 6_unknown_crop_rice | rice: not one of the 14 crops | refer (energy gate misses rice) |
| 7_not_a_leaf | not a plant at all | refer |

`expected_results.csv` lists, for every photo: the true disease and group, the demo model's
decision, its prediction set and its top guess.

IMPORTANT: the "expected" answers come from the DEMO model shipped in saved_model/
(MobileNetV3-Small, frozen, trained on CPU). If you replaced saved_model/ with your Colab
EfficientNet-B0 model, many photos in folders 2-5 will get a DIFFERENT answer. That is normal:
the folders show which situations exist; your model decides each photo again.
Folder 1 (lab photos) should still be "advise" with the right group for almost every photo.
