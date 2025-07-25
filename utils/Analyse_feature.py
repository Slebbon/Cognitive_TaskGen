"""
Feature Analysis Utilities

This module contains reusable functions for analyzing and validating features
in AI detection tasks. It provides comprehensive analysis including statistical
validation, SHAP analysis, model comparison, and feature worthiness evaluation.

Usage:
    from utils import FeatureAnalyzer
    
    analyzer = FeatureAnalyzer()
    results = analyzer.analyze_features(df, feature_names, target_col)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    classification_report, confusion_matrix, auc
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.pipeline import Pipeline
from scipy import stats
import shap
import xgboost as xgb
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')


class FeatureAnalyzer:
    """
    Comprehensive feature analysis toolkit for AI detection tasks
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def visualize_features(self, df, feature_names, target_col='binary_label_code', 
                          feature_type_name="Features"):
        """
        Create comprehensive visualizations for features
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing features and target
        feature_names : list
            List of feature column names to analyze
        target_col : str
            Target column name
        feature_type_name : str
            Name for the feature type (for plot titles)
        """
        n_features = len(feature_names)
        
        if n_features <= 3:
            # Compact layout for 3 or fewer features
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle(f'{feature_type_name} Analysis: Human vs AI-Generated Text', fontsize=16)
            
            # Distribution plots
            for i, col in enumerate(feature_names):
                ax = axes[0, i]
                
                human_data = df[df[target_col] == 0][col]
                ai_data = df[df[target_col] == 1][col]
                
                ax.hist(human_data, alpha=0.7, label='Human', bins=30, color='blue', density=True)
                ax.hist(ai_data, alpha=0.7, label='AI', bins=30, color='red', density=True)
                ax.set_title(f'{col.replace("_", " ").title()}')
                ax.legend()
                ax.set_xlabel('Value')
                ax.set_ylabel('Density')
            
            # Remove empty subplots in first row
            for i in range(n_features, 3):
                axes[0, i].remove()
            
            # Correlation heatmap
            ax = axes[1, 0]
            corr_matrix = df[feature_names + [target_col]].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax,
                       cbar_kws={'shrink': 0.8})
            ax.set_title('Feature Correlation Matrix')
            
            # Box plots
            ax = axes[1, 1]
            melted_data = df[feature_names + [target_col]].melt(id_vars=[target_col])
            sns.boxplot(data=melted_data, x='variable', y='value', hue=target_col, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            ax.set_title('Feature Distributions by Class')
            ax.legend(title='Class', labels=['Human', 'AI'])
            
            # Statistical summary table
            ax = axes[1, 2]
            stats_data = []
            for col in feature_names:
                human_data = df[df[target_col] == 0][col]
                ai_data = df[df[target_col] == 1][col]
                t_stat, p_val = stats.ttest_ind(human_data, ai_data)
                effect_size = (np.mean(human_data) - np.mean(ai_data)) / np.sqrt(
                    (np.var(human_data) + np.var(ai_data)) / 2
                )
                stats_data.append([col.replace('_', '\n'), f'{p_val:.3f}', f'{effect_size:.3f}'])
            
            table_data = [['Feature', 'p-value', 'Effect Size']] + stats_data
            ax.axis('tight')
            ax.axis('off')
            table = ax.table(cellText=table_data, cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            ax.set_title('Statistical Summary')
            
        else:
            # Extended layout for more features
            n_rows = (n_features + 2) // 3
            fig, axes = plt.subplots(n_rows + 1, 3, figsize=(18, 5 * (n_rows + 1)))
            fig.suptitle(f'{feature_type_name} Analysis: Human vs AI-Generated Text', fontsize=16)
            
            # Distribution plots
            for i, col in enumerate(feature_names):
                row, col_idx = i // 3, i % 3
                ax = axes[row, col_idx]
                
                human_data = df[df[target_col] == 0][col]
                ai_data = df[df[target_col] == 1][col]
                
                ax.hist(human_data, alpha=0.7, label='Human', bins=30, color='blue', density=True)
                ax.hist(ai_data, alpha=0.7, label='AI', bins=30, color='red', density=True)
                ax.set_title(f'{col.replace("_", " ").title()}')
                ax.legend()
                ax.set_xlabel('Value')
                ax.set_ylabel('Density')
            
            # Additional plots in last row
            last_row = n_rows
            
            # Correlation heatmap
            ax = axes[last_row, 0]
            corr_matrix = df[feature_names + [target_col]].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax,
                       cbar_kws={'shrink': 0.8})
            ax.set_title('Feature Correlation Matrix')
            
            # Box plots (first 4 features)
            ax = axes[last_row, 1]
            viz_features = feature_names[:4]
            melted_data = df[viz_features + [target_col]].melt(id_vars=[target_col])
            sns.boxplot(data=melted_data, x='variable', y='value', hue=target_col, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            ax.set_title('Feature Distributions by Class')
            ax.legend(title='Class', labels=['Human', 'AI'])
        
        plt.tight_layout()
        plt.show()
        return fig
    
    def statistical_validation(self, df, feature_names, target_col='binary_label_code'):
        """
        Comprehensive statistical validation of features
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing features and target
        feature_names : list
            List of feature column names to analyze
        target_col : str
            Target column name
            
        Returns:
        --------
        dict : Statistical validation results
        """
        print("=== STATISTICAL VALIDATION RESULTS ===\n")
        
        # T-tests for group differences
        print("1. T-TEST RESULTS (Human vs AI):")
        print("-" * 60)
        print(f"{'Feature':<25} {'t-stat':<8} {'p-value':<10} {'Effect Size':<12} {'Significant'}")
        print("-" * 60)
        
        significant_features = []
        
        for col in feature_names:
            human_data = df[df[target_col] == 0][col]
            ai_data = df[df[target_col] == 1][col]
            
            t_stat, p_value = stats.ttest_ind(human_data, ai_data)
            effect_size = (np.mean(human_data) - np.mean(ai_data)) / np.sqrt(
                (np.var(human_data) + np.var(ai_data)) / 2
            )
            
            is_significant = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            if p_value < 0.05:
                significant_features.append(col)
            
            print(f"{col:<25} {t_stat:>7.3f} {p_value:>9.4f} {effect_size:>11.3f} {is_significant:>11}")
        
        # Mutual information
        print(f"\n2. MUTUAL INFORMATION SCORES:")
        print("-" * 40)
        
        X = df[feature_names].fillna(0)
        y = df[target_col]
        
        mi_scores = mutual_info_classif(X, y, random_state=42)
        mi_results = list(zip(feature_names, mi_scores))
        mi_results.sort(key=lambda x: x[1], reverse=True)
        
        for col, score in mi_results:
            print(f"{col:<25}: {score:.4f}")
        
        # Correlation with target
        print(f"\n3. CORRELATION WITH TARGET:")
        print("-" * 40)
        
        correlations = []
        for col in feature_names:
            corr = df[col].corr(df[target_col])
            correlations.append((col, abs(corr), corr))
        
        correlations.sort(key=lambda x: x[1], reverse=True)
        
        for col, abs_corr, corr in correlations:
            print(f"{col:<25}: {corr:>7.4f}")
        
        # Summary
        print(f"\n=== SUMMARY ===")
        print(f"Significant features (p < 0.05): {len(significant_features)}/{len(feature_names)}")
        print(f"Most important by MI: {mi_results[0][0]} ({mi_results[0][1]:.4f})")
        print(f"Highest correlation: {correlations[0][0]} ({correlations[0][2]:.4f})")
        
        return {
            'significant_features': significant_features,
            'mi_scores': dict(mi_results),
            'correlations': {col: corr for col, _, corr in correlations}
        }
    
    def model_validation(self, df, feature_names, target_col='binary_label_code'):
        """
        Advanced model validation with multiple algorithms
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing features and target
        feature_names : list
            List of feature column names to analyze
        target_col : str
            Target column name
            
        Returns:
        --------
        dict : Model validation results
        """
        X = df[feature_names].fillna(0)
        y = df[target_col]
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0)
        }
        
        results = {}
        
        print("=== ADVANCED MODEL VALIDATION ===\n")
        
        for name, model in models.items():
            print(f"{name}:")
            print("-" * 40)
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='roc_auc')
            
            print(f"Test AUC: {auc_score:.4f}")
            print(f"CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Classification report
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_
                print("Feature Importance:")
                importance_pairs = list(zip(feature_names, feature_importance))
                importance_pairs.sort(key=lambda x: x[1], reverse=True)
                for col, imp in importance_pairs:
                    print(f"  {col:<25}: {imp:.4f}")
            elif hasattr(model, 'coef_'):
                feature_importance = np.abs(model.coef_[0])
                print("Feature Coefficients (absolute):")
                importance_pairs = list(zip(feature_names, feature_importance))
                importance_pairs.sort(key=lambda x: x[1], reverse=True)
                for col, coef in importance_pairs:
                    print(f"  {col:<25}: {coef:.4f}")
            
            results[name] = {
                'auc': auc_score,
                'cv_auc_mean': cv_scores.mean(),
                'cv_auc_std': cv_scores.std(),
                'model': model,
                'y_test': y_test,
                'y_pred_proba': y_pred_proba
            }
            
            print("\n" + "="*50 + "\n")
        
        return results
    
    def shap_analysis(self, df, feature_names, target_col='binary_label_code'):
        """
        SHAP analysis using XGBoost
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing features and target
        feature_names : list
            List of feature column names to analyze
        target_col : str
            Target column name
            
        Returns:
        --------
        dict : SHAP analysis results
        """
        X = df[feature_names].fillna(0)
        y = df[target_col]
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled_df, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print("=== SHAP ANALYSIS (XGBoost) ===\n")
        
        # Train XGBoost model
        model = XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0)
        model.fit(X_train, y_train)
        
        # Model performance
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
        print(f"Model AUC: {auc_score:.4f}")
        
        # SHAP analysis
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        # Handle different SHAP formats
        if isinstance(shap_values, list):
            if len(shap_values) == 2:
                shap_values_final = shap_values[1]
            else:
                shap_values_final = shap_values[0]
        else:
            shap_values_final = shap_values
        
        # Calculate feature importance
        mean_abs_shap = np.mean(np.abs(shap_values_final), axis=0)
        feature_importance_shap = dict(zip(feature_names, mean_abs_shap))
        
        print("\nMean Absolute SHAP Values (Feature Importance):")
        sorted_features = sorted(feature_importance_shap.items(), key=lambda x: x[1], reverse=True)
        for feature, importance in sorted_features:
            print(f"  {feature:<25}: {importance:.4f}")
        
        # Visualizations
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('XGBoost SHAP Analysis: Feature Importance and Impact', fontsize=16)
            
            # Feature importance bar plot
            ax1 = axes[0]
            shap.summary_plot(shap_values_final, X_test, plot_type="bar", show=False, ax=ax1)
            ax1.set_title('SHAP Feature Importance')
            
            # Beeswarm plot
            ax2 = axes[1]
            shap.summary_plot(shap_values_final, X_test, show=False, ax=ax2)
            ax2.set_title('SHAP Value Distribution')
            
            # Manual feature importance plot
            ax3 = axes[2]
            features = list(feature_importance_shap.keys())
            importances = list(feature_importance_shap.values())
            
            sorted_idx = np.argsort(importances)
            ax3.barh(range(len(features)), [importances[i] for i in sorted_idx], color='steelblue')
            ax3.set_yticks(range(len(features)))
            ax3.set_yticklabels([features[i].replace('_', ' ').title() for i in sorted_idx])
            ax3.set_xlabel('Mean |SHAP Value|')
            ax3.set_title('Feature Importance (Manual)')
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Warning: SHAP plots could not be generated: {str(e)}")
            # Fallback manual plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            features = list(feature_importance_shap.keys())
            importances = list(feature_importance_shap.values())
            
            sorted_idx = np.argsort(importances)
            ax.barh(range(len(features)), [importances[i] for i in sorted_idx], color='steelblue')
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels([features[i].replace('_', ' ').title() for i in sorted_idx])
            ax.set_xlabel('Mean |SHAP Value|')
            ax.set_title('SHAP Feature Importance')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        
        return {
            'shap_values': shap_values_final,
            'feature_importance': feature_importance_shap,
            'model': model,
            'X_test': X_test,
            'auc_score': auc_score
        }
    
    def plot_roc_pr_curves(self, model_results):
        """
        Plot ROC and Precision-Recall curves for multiple models
        
        Parameters:
        -----------
        model_results : dict
            Results from model_validation function
        """
        plt.figure(figsize=(15, 5))
        
        # ROC Curve
        plt.subplot(1, 3, 1)
        for name, result in model_results.items():
            y_test = result['y_test']
            y_pred_proba = result['y_pred_proba']
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        plt.subplot(1, 3, 2)
        for name, result in model_results.items():
            y_test = result['y_test']
            y_pred_proba = result['y_pred_proba']
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, label=f'{name} (AUC = {pr_auc:.3f})', linewidth=2)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Feature Importance (if Random Forest available)
        plt.subplot(1, 3, 3)
        if 'Random Forest' in model_results:
            rf_model = model_results['Random Forest']['model']
            if hasattr(rf_model, 'feature_importances_'):
                importance = rf_model.feature_importances_
                indices = np.argsort(importance)[::-1]
                feature_names = [f"Feature {i+1}" for i in range(len(importance))]
                
                plt.barh(range(len(importance)), importance[indices])
                plt.yticks(range(len(importance)), [feature_names[i] for i in indices])
                plt.xlabel('Feature Importance')
                plt.title('Random Forest Feature Importance')
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def evaluate_feature_worthiness(self, df, feature_names, stats_results, model_results, 
                                   shap_results=None, target_col='binary_label_code'):
        """
        Comprehensive feature worthiness evaluation
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing features and target
        feature_names : list
            List of feature column names to analyze
        stats_results : dict
            Results from statistical_validation
        model_results : dict
            Results from model_validation
        shap_results : dict, optional
            Results from shap_analysis
        target_col : str
            Target column name
            
        Returns:
        --------
        tuple : (worthiness_scores, recommendations)
        """
        print("=== FEATURE WORTHINESS EVALUATION ===\n")
        
        worthiness_scores = {}
        
        for feature in feature_names:
            score = 0
            criteria_met = []
            
            # 1. Statistical Significance
            human_data = df[df[target_col] == 0][feature]
            ai_data = df[df[target_col] == 1][feature]
            _, p_value = stats.ttest_ind(human_data, ai_data)
            
            if p_value < 0.05:
                score += 2
                criteria_met.append("Statistically Significant")
            elif p_value < 0.1:
                score += 1
                criteria_met.append("Marginally Significant")
            
            # 2. Effect Size
            effect_size = (np.mean(human_data) - np.mean(ai_data)) / np.sqrt(
                (np.var(human_data) + np.var(ai_data)) / 2
            )
            
            if abs(effect_size) > 0.8:
                score += 3
                criteria_met.append("Large Effect Size")
            elif abs(effect_size) > 0.5:
                score += 2
                criteria_met.append("Medium Effect Size")
            elif abs(effect_size) > 0.2:
                score += 1
                criteria_met.append("Small Effect Size")
            
            # 3. Correlation with target
            correlation = abs(df[feature].corr(df[target_col]))
            
            if correlation > 0.5:
                score += 3
                criteria_met.append("Strong Correlation")
            elif correlation > 0.3:
                score += 2
                criteria_met.append("Moderate Correlation")
            elif correlation > 0.1:
                score += 1
                criteria_met.append("Weak Correlation")
            
            # 4. Mutual Information
            if 'mi_scores' in stats_results:
                mi_score = stats_results['mi_scores'].get(feature, 0)
                max_mi = max(stats_results['mi_scores'].values())
                normalized_mi = mi_score / max_mi if max_mi > 0 else 0
                
                if normalized_mi > 0.7:
                    score += 2
                    criteria_met.append("High Mutual Information")
                elif normalized_mi > 0.4:
                    score += 1
                    criteria_met.append("Moderate Mutual Information")
            
            # 5. SHAP Importance
            if shap_results and 'feature_importance' in shap_results:
                shap_importance = shap_results['feature_importance'].get(feature, 0)
                max_shap = max(shap_results['feature_importance'].values())
                normalized_shap = shap_importance / max_shap if max_shap > 0 else 0
                
                if normalized_shap > 0.7:
                    score += 2
                    criteria_met.append("High SHAP Importance")
                elif normalized_shap > 0.4:
                    score += 1
                    criteria_met.append("Moderate SHAP Importance")
            
            worthiness_scores[feature] = {
                'score': score,
                'criteria_met': criteria_met,
                'p_value': p_value,
                'effect_size': effect_size,
                'correlation': correlation
            }
        
        # Create recommendations
        print("FEATURE WORTHINESS SCORES:")
        print("-" * 80)
        print(f"{'Feature':<25} {'Score':<6} {'Recommendation':<15} {'Key Criteria'}")
        print("-" * 80)
        
        recommendations = {}
        
        for feature, data in sorted(worthiness_scores.items(), key=lambda x: x[1]['score'], reverse=True):
            score = data['score']
            
            if score >= 8:
                recommendation = "MUST INCLUDE"
            elif score >= 5:
                recommendation = "INCLUDE"
            elif score >= 3:
                recommendation = "CONSIDER"
            else:
                recommendation = "EXCLUDE"
            
            key_criteria = ", ".join(data['criteria_met'][:2])
            
            print(f"{feature:<25} {score:<6} {recommendation:<15} {key_criteria}")
            recommendations[feature] = recommendation
        
        # Summary
        must_include = [f for f, r in recommendations.items() if r == "MUST INCLUDE"]
        include = [f for f, r in recommendations.items() if r == "INCLUDE"]
        consider = [f for f, r in recommendations.items() if r == "CONSIDER"]
        exclude = [f for f, r in recommendations.items() if r == "EXCLUDE"]
        
        print(f"\n=== SUMMARY ===")
        print(f"✅ MUST INCLUDE ({len(must_include)}): {', '.join(must_include)}")
        print(f"✅ INCLUDE ({len(include)}): {', '.join(include)}")
        print(f"⚠️  CONSIDER ({len(consider)}): {', '.join(consider)}")
        print(f"❌ EXCLUDE ({len(exclude)}): {', '.join(exclude)}")
        
        return worthiness_scores, recommendations
    
    def compare_with_existing_features(self, df, new_features, existing_features, 
                                     target_col='binary_label_code'):
        """
        Compare new features against existing features using SHAP and AUC
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing all features
        new_features : list
            List of new feature names
        existing_features : list
            List of existing feature names
        target_col : str
            Target column name
            
        Returns:
        --------
        dict : Comparison results
        """
        print("=== FEATURE COMPARISON ANALYSIS ===\n")
        
        # Prepare data
        X_new = df[new_features].fillna(0)
        X_existing = df[existing_features].fillna(0)
        X_combined = df[new_features + existing_features].fillna(0)
        y = df[target_col]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Feature sets for comparison
        X_train_new = X_train[new_features]
        X_test_new = X_test[new_features]
        X_train_existing = X_train[existing_features]
        X_test_existing = X_test[existing_features]
        X_train_combined = X_train[new_features + existing_features]
        X_test_combined = X_test[new_features + existing_features]
        
        # Logistic Regression Comparison
        print("LOGISTIC REGRESSION COMPARISON:")
        print("-" * 50)
        
        # Pipelines
        pipe_new = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, random_state=42))
        ])
        
        pipe_existing = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, random_state=42))
        ])
        
        pipe_combined = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, random_state=42))
        ])
        
        # Fit and evaluate
        pipe_new.fit(X_train_new, y_train)
        pipe_existing.fit(X_train_existing, y_train)
        pipe_combined.fit(X_train_combined, y_train)
        
        auc_new = roc_auc_score(y_test, pipe_new.predict_proba(X_test_new)[:, 1])
        auc_existing = roc_auc_score(y_test, pipe_existing.predict_proba(X_test_existing)[:, 1])
        auc_combined = roc_auc_score(y_test, pipe_combined.predict_proba(X_test_combined)[:, 1])
        
        print(f"ROC-AUC new features only: {auc_new:.3f}")
        print(f"ROC-AUC existing features only: {auc_existing:.3f}")
        print(f"ROC-AUC combined features: {auc_combined:.3f}")
        print(f"Improvement: +{auc_combined - auc_existing:.3f}")
        
        # XGBoost + SHAP Analysis
        print(f"\nXGBOOST + SHAP ANALYSIS:")
        print("-" * 50)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train_combined),
            columns=X_train_combined.columns,
            index=X_train_combined.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test_combined),
            columns=X_test_combined.columns,
            index=X_test_combined.index
        )
        
        # Train XGBoost
        model_xgb = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            n_estimators=300,
            random_state=42
        )
        model_xgb.fit(X_train_scaled, y_train)
        
        auc_xgb = roc_auc_score(y_test, model_xgb.predict_proba(X_test_scaled)[:, 1])
        print(f"XGBoost AUC: {auc_xgb:.3f}")
        
        # SHAP Analysis
        explainer = shap.Explainer(model_xgb, X_train_scaled)
        shap_values = explainer(X_test_scaled)
        
        # Feature importance
        feature_importance = np.abs(shap_values.values).mean(axis=0)
        feature_names = X_test_scaled.columns
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance,
            'type': ['New' if f in new_features else 'Existing' for f in feature_names]
        })
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Display results
        print("\nTop 10 Features:")
        for i, row in importance_df.head(10).iterrows():
            feature_type = "🔴" if row['type'] == 'New' else "🔵"
            print(f"{feature_type} {row['feature']:30s} | {row['importance']:.4f}")
        
        # New features ranking
        print(f"\nNew Features Performance:")
        for feature in new_features:
            if feature in importance_df['feature'].values:
                rank = importance_df[importance_df['feature'] == feature].index[0] + 1
                importance = importance_df[importance_df['feature'] == feature]['importance'].values[0]
                print(f"  {feature:25s} | Rank #{rank:2d} | SHAP: {importance:.4f}")
        
        # Summary statistics
        new_importances = importance_df[importance_df['type'] == 'New']['importance']
        existing_importances = importance_df[importance_df['type'] == 'Existing']['importance']
        
        print(f"\nSummary:")
        print(f"New features in top 10: {sum(importance_df.head(10)['type'] == 'New')}/{len(new_features)}")
        print(f"New features mean importance: {new_importances.mean():.4f}")
        print(f"Existing features mean importance: {existing_importances.mean():.4f}")
        
        # Generate plots
        shap.plots.bar(shap_values, max_display=15)
        
        # Recommendation
        improvement = auc_combined - auc_existing
        if improvement > 0.02:
            recommendation = "🟢 STRONG: Add new features"
        elif improvement > 0.01:
            recommendation = "🟡 MODERATE: Consider adding new features"
        elif improvement > 0.005:
            recommendation = "🟠 WEAK: Limited benefit from new features"
        else:
            recommendation = "🔴 MINIMAL: New features add little value"
        
        print(f"\n💡 Recommendation: {recommendation}")
        
        return {
            'auc_new': auc_new,
            'auc_existing': auc_existing,
            'auc_combined': auc_combined,
            'improvement': improvement,
            'importance_df': importance_df,
            'recommendation': recommendation
        }
    
    def analyze_features(self, df, feature_names, target_col='binary_label_code', 
                        feature_type_name="Features", existing_features=None):
        """
        Complete feature analysis pipeline
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing features and target
        feature_names : list
            List of feature column names to analyze
        target_col : str
            Target column name
        feature_type_name : str
            Name for the feature type
        existing_features : list, optional
            List of existing features for comparison
            
        Returns:
        --------
        dict : Complete analysis results
        """
        print(f"Starting {feature_type_name} Analysis Pipeline...")
        print("=" * 60)
        
        # Step 1: Visualizations
        print(f"\nStep 1: Creating visualizations...")
        self.visualize_features(df, feature_names, target_col, feature_type_name)
        
        # Step 2: Statistical validation
        print(f"\nStep 2: Statistical validation...")
        stats_results = self.statistical_validation(df, feature_names, target_col)
        
        # Step 3: Model validation
        print(f"\nStep 3: Model validation...")
        model_results = self.model_validation(df, feature_names, target_col)
        
        # Step 4: SHAP analysis
        print(f"\nStep 4: SHAP analysis...")
        shap_results = self.shap_analysis(df, feature_names, target_col)
        
        # Step 5: Feature worthiness
        print(f"\nStep 5: Feature worthiness evaluation...")
        worthiness_scores, recommendations = self.evaluate_feature_worthiness(
            df, feature_names, stats_results, model_results, shap_results, target_col
        )
        
        # Step 6: ROC curves
        print(f"\nStep 6: ROC and PR curves...")
        self.plot_roc_pr_curves(model_results)
        
        # Step 7: Comparison with existing features (if provided)
        comparison_results = None
        if existing_features:
            print(f"\nStep 7: Comparison with existing features...")
            comparison_results = self.compare_with_existing_features(
                df, feature_names, existing_features, target_col
            )
        
        print(f"\n{feature_type_name} analysis complete!")
        
        return {
            'stats_results': stats_results,
            'model_results': model_results,
            'shap_results': shap_results,
            'worthiness_scores': worthiness_scores,
            'recommendations': recommendations,
            'comparison_results': comparison_results
        }


def merge_features_to_aggregated(df_with_features, feature_names, aggregated_path, 
                                id_column='id', save_path=None):
    """
    Utility function to merge new features into aggregated dataframe
    
    Parameters:
    -----------
    df_with_features : pd.DataFrame
        DataFrame containing the new features
    feature_names : list
        List of feature names to merge
    aggregated_path : str
        Path to the aggregated dataframe
    id_column : str
        ID column name for merging
    save_path : str, optional
        Path to save updated aggregated dataframe
    
    Returns:
    --------
    pd.DataFrame : Updated aggregated dataframe
    """
    import os
    
    # Load or create aggregated dataframe
    if os.path.exists(aggregated_path):
        aggregated_df = pd.read_csv(aggregated_path)
        print("Loaded existing aggregated_df.")
    else:
        print("No existing aggregated_df found. Creating new one.")
        aggregated_df = df_with_features[[id_column]].copy()
    
    # Extract and merge new features
    feature_df = df_with_features[[id_column] + feature_names].copy()
    aggregated_df = aggregated_df.merge(feature_df, on=id_column, how="left")
    
    # Save
    save_path = save_path or aggregated_path
    aggregated_df.to_csv(save_path, index=False)
    print(f"Updated aggregated_df saved to: {save_path}")
    
    return aggregated_df