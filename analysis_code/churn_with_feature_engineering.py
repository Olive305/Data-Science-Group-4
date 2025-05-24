import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class CausalChurnModel:
    
    def __init__(self):
        self.churn_model = None
        self.profit_model = None
        self.scaler_churn = StandardScaler()
        self.scaler_profit = StandardScaler()
        self.feature_names = []
        self.satisfaction_features = []
        self.financial_features = []
        self.model_features = []
        
    def load_and_prepare_data(self, full_data_path):
        """
        Load the full_data.xlsx and prepare it for causal modeling
        """
        print("Loading data...")
        df = pd.read_excel(full_data_path)
        
        # Convert string columns to numeric where possible
        satisfaction_cols = [
            'Globalzufriedenheit (Schulnote)', 'Preis-Leistungs-Verhältnis (Schulnote)',
            'Wiederwahlabsicht (Schulnote)', 'Weiterempfehlungsabsicht (Schulnote)',
            'Weiterempfehlungsabsicht (NPS)', 'Kundenloyalität (Schulnote)',
            'Leistungsumfang', 'Verständlichkeit der schriftlichen Unterlagen',
            'Informationen zu Gesundheitsthemen', 'Umfang der Online-Services',
            'Nutzerfreundlichkeit der Online-Services', 'Individuelle Beratung',
            'Bonusprogramm', 'Wahltarife', 'Präventionsangebote',
            'Vermittlung von Arztterminen', 'Medizinische Hotline',
            'Persönl. Kundenbereich', 'App der Krankenkasse',
            'Video- oder Online-Beratung', 'Newsletter',
            'Geschäftsstelle Erreichbarkeit', 'Geschäftsstelle Öffnungszeiten',
            'Geschäftsstelle Wartezeiten', 'Geschäftsstelle Erscheinungsbild',
            'Geschäftsstelle Freundlichkeit der Mitarbeiter',
            'Geschäftsstelle Fachliche Beratung', 'Geschäftsstelle Erledigung des Anliegens',
            'Telefonische Erreichbarkeit', 'Telefon Wartezeiten',
            'Telefon Freundlichkeit', 'Telefon Fachliche Beratung',
            'Telefon Erledigung des Anliegens',
            'Online-Kontakt Erledigung der online übermittelten Anfragen',
            'Aktive Betreuung', 'Persönlicher Ansprechpartner',
            'Kundenmagazin Gesamtzufriedenheit'
        ]
        
        # Convert satisfaction columns to numeric
        for col in satisfaction_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Ensure numeric columns are properly typed
        numeric_cols = ['Jahr', 'Quartal', 'Mitglieder', 'Versicherte', 'Zusatzbeitrag', 'Risikofaktor']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"Loaded {len(df)} records for {df['Krankenkasse'].nunique()} insurance companies")
        return df
        
    def engineer_features(self, df):
        """
        Create engineered features for causal modeling
        """
        print("Engineering features...")
        data = df.copy()
        
        data = data.dropna(subset=['Jahr', 'Quartal'])
        data['Jahr'] = data['Jahr'].astype(int)
        data['Quartal'] = data['Quartal'].astype(int)
        # Create time period for sorting
        data['Date'] = pd.PeriodIndex.from_fields(year=data['Jahr'], quarter=data['Quartal'], freq='Q')
        data = data.sort_values(by=['Krankenkasse', 'Date']).reset_index(drop=True)
        
        # Fee-related features
        data['Zusatzbeitrag_diff'] = data.groupby('Krankenkasse')['Zusatzbeitrag'].diff()
        data['Zusatzbeitrag_pct_change'] = data.groupby('Krankenkasse')['Zusatzbeitrag'].pct_change() * 100
        
        # Member changes (our main target for churn prediction)
        data['Mitglieder_change_next'] = data.groupby('Krankenkasse')['Mitglieder'].diff(periods=-1) * -1  # Next quarter change
        data['Mitglieder_pct_change_next'] = data.groupby('Krankenkasse')['Mitglieder'].pct_change(periods=-1) * -100  # Next quarter % change
        
        # Market context features
        data['Market_fee_avg'] = data.groupby('Date')['Zusatzbeitrag'].transform('mean')
        data['Fee_vs_market'] = data['Zusatzbeitrag'] - data['Market_fee_avg']
        data['Market_share'] = data.groupby('Date')['Mitglieder'].transform(lambda x: x / x.sum() * 100)
        
        # Satisfaction context
        if 'Globalzufriedenheit (Schulnote)' in data.columns:
            data['Market_satisfaction_avg'] = data.groupby('Date')['Globalzufriedenheit (Schulnote)'].transform('mean')
            data['Satisfaction_vs_market'] = data['Globalzufriedenheit (Schulnote)'] - data['Market_satisfaction_avg']
        
        # Risk and demographic features
        data['Family_ratio'] = data['Versicherte'] / data['Mitglieder']
        data['Risk_adjusted_size'] = data['Mitglieder'] * data['Risikofaktor']
        
        # Revenue and profit proxies
        data['Revenue_per_member'] = data['Zusatzbeitrag'] * 4  # Quarterly to annual
        data['Total_revenue'] = data['Revenue_per_member'] * data['Mitglieder']
        data['Revenue_change_next'] = data.groupby('Krankenkasse')['Total_revenue'].diff(periods=-1) * -1
        
        # Interaction terms (key for causal modeling)
        if 'Globalzufriedenheit (Schulnote)' in data.columns:
            data['Fee_satisfaction_interaction'] = data['Zusatzbeitrag_diff'] * data['Globalzufriedenheit (Schulnote)']
        if 'Preis-Leistungs-Verhältnis (Schulnote)' in data.columns:
            data['Fee_value_interaction'] = data['Zusatzbeitrag'] * data['Preis-Leistungs-Verhältnis (Schulnote)']
        
        # Lag features for better causal inference
        key_satisfaction_features = [
            'Globalzufriedenheit (Schulnote)', 'Preis-Leistungs-Verhältnis (Schulnote)',
            'Wiederwahlabsicht (Schulnote)', 'Kundenloyalität (Schulnote)'
        ]
        
        for feature in key_satisfaction_features:
            if feature in data.columns:
                data[f'{feature}_lag1'] = data.groupby('Krankenkasse')[feature].shift(1)
        
        print(f"Created features. Data shape: {data.shape}")
        return data
    
    def select_model_features(self, data):
        """
        Select the most relevant features for modeling
        """
        # Core satisfaction features (most predictive)
        core_satisfaction = [
            'Globalzufriedenheit (Schulnote)', 'Preis-Leistungs-Verhältnis (Schulnote)',
            'Wiederwahlabsicht (Schulnote)', 'Kundenloyalität (Schulnote)',
            'Telefonische Erreichbarkeit', 'Online-Kontakt Erledigung der online übermittelten Anfragen',
            'Geschäftsstelle Freundlichkeit der Mitarbeiter'
        ]
        
        # Financial and market features
        financial_features = [
            'Zusatzbeitrag_diff', 'Zusatzbeitrag_pct_change', 'Fee_vs_market',
            'Market_share', 'Risikofaktor', 'Family_ratio'
        ]
        
        # Interaction and context features
        interaction_features = [
            'Fee_satisfaction_interaction', 'Fee_value_interaction', 'Satisfaction_vs_market'
        ]
        
        # Select available features
        all_potential_features = core_satisfaction + financial_features + interaction_features
        available_features = [f for f in all_potential_features if f in data.columns]
        
        print(f"Selected {len(available_features)} features for modeling:")
        for f in available_features:
            print(f"  - {f}")
        
        return available_features
    
    def train_models(self, data):
        """
        Train both churn and profit prediction models
        """
        # Select features
        self.model_features = self.select_model_features(data)
        
        # Remove rows with missing target variables or key features
        model_data = data.dropna(subset=['Mitglieder_pct_change_next'] + self.model_features[:8])  # Use top features
        
        print(f"Training on {len(model_data)} samples")
        
        if len(model_data) < 30:
            print("Warning: Very limited training data. Results may be unreliable.")
            return False
        
        # Prepare churn model data
        X = model_data[self.model_features[:8]]  # Limit to prevent overfitting
        y_churn = model_data['Mitglieder_pct_change_next']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y_churn, test_size=0.25, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler_churn.fit_transform(X_train)
        X_test_scaled = self.scaler_churn.transform(X_test)
        
        # Train churn model with cross-validation
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42),
            'Ridge Regression': Ridge(alpha=1.0)
        }
        
        best_score = -np.inf
        best_model_name = None
        
        print("\nChurn Model Training Results:")
        print("-" * 40)
        
        for name, model in models.items():
            try:
                # Cross-validation
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring='r2')
                avg_score = np.mean(cv_scores)
                
                print(f"{name}: CV R² = {avg_score:.4f} ± {np.std(cv_scores):.4f}")
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_model_name = name
                    self.churn_model = model
            except Exception as e:
                print(f"{name}: Failed to train - {str(e)}")
        
        if self.churn_model is not None:
            # Train best model
            self.churn_model.fit(X_train_scaled, y_train)
            
            # Test performance
            y_pred = self.churn_model.predict(X_test_scaled)
            test_r2 = r2_score(y_test, y_pred)
            test_mse = mean_squared_error(y_test, y_pred)
            
            print(f"\nBest Model ({best_model_name}):")
            print(f"  Test R² Score: {test_r2:.4f}")
            print(f"  Test MSE: {test_mse:.4f}")
            
            # Feature importance
            if hasattr(self.churn_model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': self.churn_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                print(f"\nTop Feature Importances:")
                for _, row in feature_importance.head().iterrows():
                    print(f"  {row['feature']}: {row['importance']:.3f}")
        
        # Train simple profit model if revenue data available
        if 'Revenue_change_next' in model_data.columns:
            profit_data = model_data.dropna(subset=['Revenue_change_next'])
            if len(profit_data) > 20:
                X_profit = profit_data[self.model_features[:8]]
                y_profit = profit_data['Revenue_change_next']
                
                self.profit_model = Ridge(alpha=1.0)
                X_profit_scaled = self.scaler_profit.fit_transform(X_profit)
                self.profit_model.fit(X_profit_scaled, y_profit)
                print(f"\nProfit model trained on {len(profit_data)} samples")
        
        return True

# Main execution function
def run_complete_analysis(full_data_path, company_name=None):
    """
    Run the complete causal analysis pipeline
    """
    print("Starting Causal Churn Analysis...")
    
    # Initialize model
    model = CausalChurnModel()
    
    # Load and prepare data
    df = model.load_and_prepare_data(full_data_path)
    engineered_df = model.engineer_features(df)
    
    # Train models
    model.train_models(engineered_df)
    

    
analysis = run_complete_analysis('data/custom_files/full_data.xlsx')