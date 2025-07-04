


def main():
    while True:
        print("Choose what you want to do : command\n"
              "Exploratory Data Analysis : eda\n"
              "Linear Regression : lr\n"
              "Neural Network : nn\n"
              "Causal Forest : cf\n"
              "Mixed Effects Model : mem\n"
              "Difference in Difference : did\n"
              "Panel Regression : plr\n"
              "Monte Carlo for all insurers based on CF: mcf\n" 
              "Explore the predictions in the webapp: webapp"
              "exit : exit\n"
              "(Most models other than nn are currently trained when called this will change if all of the data is"
              "included but right now this makes sense for testing)"
              )
        choice = input("Enter: ").strip().lower()
        if choice == "eda":
            from data_extraction.exploratory_data_analysis import full_eda
            full_eda()
        elif choice == 'mcf':
            from paper.test import full_monte
            full_monte()
        elif choice == "lr":
            from analysis_code.linear_regression import regression_fm_adj_r2
            regression_fm_adj_r2(cv=5)
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
        elif choice == "cf":
            from analysis_code.cf_honest_trees import run_causal_forest_crossfit
            run_causal_forest_crossfit()
        elif choice == "plr":
            from analysis_code.difference_in_difference import panel
            print(panel().summary())
        elif choice == "mem":
            from analysis_code.mixed_effects_model import mem
            mem()
        elif choice == "did":
            from analysis_code.difference_in_difference import did
            print(did().summary())
        elif choice == "webapp":
            from web_interface_dashboard.interface import webapp
            webapp()
        elif choice == "exit":
            break
        else:
            print("invalid choice -> try again")



if __name__ == "__main__":
    main()
