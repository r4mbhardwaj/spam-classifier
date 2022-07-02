import csv
import os
rows = []
files = []
path_ = "spam_filter/YouTube-Spam-Collection-v1"
files.extend(os.listdir(path_))

for file_id in files:
    if file_id.endswith('csv'):
        with open(os.path.join(path_, file_id), 'r') as file:
            csvreader = csv.reader(file)
            header = next(csvreader)
            for row in csvreader:
                rows.append(row)

# \u2026 is a unicode character for ellipsis

# function that convert unicode character to text
def convert_unicode(text):
    return text.replace('\u2026', '...')

training_sentences = []
training_labels = []

labels = []

for row in rows:
    training_sentences.append(row[header.index("CONTENT")])
    training_labels.append(row[header.index("CLASS")])
    if row[header.index("CLASS")] not in labels:
        labels.append(row[header.index("CLASS")])

for index, a in enumerate(training_sentences):
    print(a, training_labels[index])
    
    
num_classes = len(labels)
print(f"num of classes: {num_classes}")