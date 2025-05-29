


def main():
    while True:
        print("Choose what you want to do : command\n"
              "Exploratory Data Analysis : eda\n"
              "Linear Regression : lr\n"
              "Neural Network : nn\n"
              "Random Forest : rf\n"
              "Mixed Effects Model : mem\n"
              "Difference in Difference : did"
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
            print("not available")
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
