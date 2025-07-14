import gradio as gr
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load & clean data
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
df = df.dropna(subset=['Age', 'Embarked'])
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3}).fillna(4)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
df['FarePerPerson'] = df['Fare'] / df['FamilySize']

features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'FamilySize', 'IsAlone', 'FarePerPerson']
X = df[features]
y = df['Survived']

model = RandomForestClassifier()
model.fit(X, y)

def predict_survival(Pclass, Sex, Age, Fare, Embarked, Title, FamilySize, IsAlone, FarePerPerson):
    input_data = pd.DataFrame([[Pclass, Sex, Age, Fare, Embarked, Title, FamilySize, IsAlone, FarePerPerson]],
                              columns=features)
    pred = model.predict(input_data)[0]
    return "✅ Survived" if pred == 1 else "❌ Did Not Survive"

demo = gr.Interface(
    fn=predict_survival,
    inputs=[
        gr.Number(label="Pclass (1-3)"),
        gr.Radio([0, 1], label="Sex (0 = Male, 1 = Female)"),
        gr.Number(label="Age"),
        gr.Number(label="Fare"),
        gr.Dropdown([0, 1, 2], label="Embarked (0 = S, 1 = C, 2 = Q)"),
        gr.Dropdown([0, 1, 2, 3, 4], label="Title (0=Mr,1=Miss,2=Mrs,3=Master,4=Rare)"),
        gr.Number(label="FamilySize"),
        gr.Radio([0, 1], label="IsAlone"),
        gr.Number(label="FarePerPerson")
    ],
    outputs="text",
    title="🚢 Titanic Survival Predictor"
)

demo.launch()
