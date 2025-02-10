import Train_LR
import Train_NB
import preprocess_data


LR = Train_LR.TrainLR("data/processed_data.csv")
NB = Train_NB.TrainNB("data/processed_data.csv")
ppd = preprocess_data.PreprocessData()

while (1):
    function = input("Enter the function you want to run: ")

    if function == "lr":
        LR.main()
    elif function == "nb":
        NB.main()
    elif function == "ppd":
        ppd.preprocess_trainingdata(
            "data/training.1600000.processed.noemoticon.csv")
    elif function == "exit" or function == "":
        break
