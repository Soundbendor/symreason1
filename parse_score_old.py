import os
import ms3
DS_FOLDER = os.path.expanduser("~/ds")
MOZ_FOLDER = os.path.join(DS_FOLDER, "mozart_piano_sonatas-main")


labels = ms3.load_tsv(os.path.join(MOZ_FOLDER, "harmonies/K279-1.harmonies.tsv"))
notes = ms3.load_tsv(os.path.join(MOZ_FOLDER, "notes/K279-1.notes.tsv"))

print(labels.dtypes)
#print(labels['duration_qb'])
