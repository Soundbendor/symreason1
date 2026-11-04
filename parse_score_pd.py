import os
import ms3_port as ms3p
import pandas as pd
DS_FOLDER = os.path.expanduser("~/ds")
MOZ_FOLDER = os.path.join(DS_FOLDER, "mozart_piano_sonatas-main")


labels = ms3p.load_tsv(os.path.join(MOZ_FOLDER, "harmonies/K279-1.harmonies.tsv"))
notes = ms3p.load_tsv(os.path.join(MOZ_FOLDER, "notes/K279-1.notes.tsv"))
metadata = pd.read_csv(os.path.join(MOZ_FOLDER, "metadata.tsv"), sep='\t' )

#print(labels.dtypes)


