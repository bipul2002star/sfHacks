from flask import Flask, render_template, request
app = Flask(__name__)
import pickle


# open a file, where you stored the pickled data
file = open('model.pkl', 'rb')
clf = pickle.load(file)
file.close()

@app.route('/', methods=["GET", "POST"])
def hello_world():
    if request.method == "POST":
        myDict = request.form
        sadness = int(myDict['sadness'])
        angryOutbursts = int(myDict['angryOutbursts'])
        lossOfInterest = int(myDict['lossOfInterest'])
        sleepIssues = int(myDict['sleepIssues'])
        tiredness = int(myDict['tiredness'])
        weightLoss = int(myDict['weightLoss'])
        anxiety = int(myDict['anxiety'])
        slowedThinking = int(myDict['slowedThinking'])
        feelingOfWorthlessness = int(myDict['feelingOfWorthlessness'])
        troubleThink = int(myDict['troubleThink'])
        suicidalThoughts = int(myDict['suicidalThoughts'])
        physicalProblems = int(myDict['physicalProblems'])
        age= int(myDict['age'])
        # Code for inference
        inputFeatures = [sadness, angryOutbursts, lossOfInterest, sleepIssues, tiredness, weightLoss, anxiety, slowedThinking, feelingOfWorthlessness, troubleThink, suicidalThoughts, physicalProblems,age]
        depressionProb =clf.predict_proba([inputFeatures])[0][1]
        print(depressionProb)
        return render_template('show.html', inf=round(depressionProb*100))
    return render_template('index.html')
 

if __name__ == "__main__":
    app.run(debug=True)