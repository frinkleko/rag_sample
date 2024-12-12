import pandas as pd
import json

l = json.load(open("qa.json", 'r', encoding='utf-8'))
df = pd.DataFrame(l).rename(columns={
    "question": "Question",
    "answer": "Reference Answer",
    "reference": "Passage",
    "file_name": "Document",
    "page": "Page",
})
# df.to_excel("qa.xlsx", index=False)
series = df.iloc[0]
print(series)
print(series[['Question', 'Passage']]['Passage'])