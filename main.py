


def main():
    while True:
        print("Choose what you want to do : command\n"
              "Exploratory Data Analysis : eda\n"
              "Linear Regression : lr\n"
              "Neural Network : nn\n"
              "Random Forest : rf\n"
              "Mixed Effects Model : mem\n"
              "Difference in Difference : did\n"
              "exit : exit"
              )
        choice = input("Enter: ").strip().lower()
        if choice == "eda":
            from data_extraction.exploratory_data_analysis import full_eda
            full_eda()
        elif choice == "lr":
            from analysis_code.predictive_models import regression_fm
            regression_fm()
        elif choice == "nn":
            from analysis_code.full_neural_network import train_and_save_neural_network, predict_from_excel
            print("Neural Network options:\n"
                "1. Train neural network\n"
                "2. Predict using neural network")
            nn_choice = input("Enter 1 or 2: ").strip()
            if nn_choice == "1":
                train_and_save_neural_network()
            elif nn_choice == "2":
                file_choice = input("Enter path to .xlsx file for prediction (leave blank to use example): ").strip()
                if file_choice:
                    predict_from_excel(file_choice)
                else:
                    predict_from_excel("./data/full_data.xlsx")
            else:
                print("Invalid neural network option")
        elif choice == "rf":
            from analysis_code.predictive_models import random_forest_regression
            random_forest_regression()
        elif choice == "mem":
            from analysis_code.mixed_effects_model import mem
            mem()
        elif choice == "did":
            print("not available")
        elif choice == "exit":
            break
        else:
            print("invalid choice -> try again")



if __name__ == "__main__":
    main()
