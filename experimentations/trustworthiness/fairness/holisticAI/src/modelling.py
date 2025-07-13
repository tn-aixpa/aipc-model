from artifact_types import Data, Configuration, Report
from interpret.blackbox import LimeTabular
from interpret import show


class Modelling():
    def __init__(self):
        self.model = None
    
    def train_model(self, data, config):
        """
        
        """
        # Split the data into training and testing sets (70% training, 30% testing)
        data_train, data_test = train_test_split(data, test_size=0.3, random_state=4)

        # Get the feature matrix (X), target labels (y), and demographic data for both sets
        X_train, y_train, dem_train = split_data_from_df(data_train)
        X_test, y_test, dem_test = split_data_from_df(data_test)

        # Define the model (RidgeClassifier) and train it on the training data
        self.model = RidgeClassifier(random_state=42)
        self.model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred_test = self.model.predict(X_test)

        # Calculate and print the accuracy of the model on the test set
        acc = accuracy_score(y_test, y_pred_test)
        print("Accuracy = %.2f" % acc)

        # Add the model's predictions to the data_test DataFrame for easier analysis
        data_test = data_test.copy()
        data_test['Pred'] = y_pred_test
        #TODO generate report to return from this method 
        return acc

############################################################## Evaluations - Performance metrics - Accuracy

    # Function to calculate and return model accuracy and fairness metrics for two groups
    def get_metrics(self, group_a, group_b, y_pred, y_true):
        """
        Returns a DataFrame of model accuracy and fairness metrics for two groups.
        """
        metrics = [['Model Accuracy', round(accuracy_score(y_true, y_pred), 2), 1]]  # Calculate accuracy
        metrics += [['Black vs. White Disparate Impact', round(disparate_impact(group_a, group_b, y_pred), 2), 1]]  # Calculate disparate impact
        metrics += [['Black vs. White Statistical Parity', round(statistical_parity(group_a, group_b, y_pred), 2), 0]]  # Calculate statistical parity
        metrics += [['Black vs. White Average Odds Difference', round(average_odds_diff(group_a, group_b, y_pred, y_true), 2), 0]]  # Calculate average odds difference
        return pd.DataFrame(metrics, columns=['Metric', 'Value', 'Reference'])  # Return metrics as DataFrame
    
    def plot_cm(self, y_true, y_pred, labels=[1, 0], display_labels=[1, 0], ax=None):
        """
        Plots a single confusion matrix with annotations
        """
        cm = confusion_matrix(y_true, y_pred, labels=labels)  # Compute confusion matrix

        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3))  # Create new figure if no axis is provided

        # Create heatmap for confusion matrix
        sns.heatmap(
            cm, annot=True, fmt="g", cmap="viridis", cbar=False,
            xticklabels=display_labels, yticklabels=display_labels,
            square=True, linewidths=2, linecolor="black", ax=ax, annot_kws={"size": 14}
        )

        # Label and format axes
        ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
        ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
        ax.set_xticklabels(display_labels, fontsize=11)
        ax.set_yticklabels(display_labels, fontsize=11)

        return cm  # Return confusion matrix


    def plot_confusion_matrices(self, groups, data_test, category, y_test, y_pred_test):
        """
        Plots confusion matrices for each group in a given category.
        """
        num_groups = len(groups) + 1  # Number of groups to display
        fig, axes = plt.subplots(1, num_groups, figsize=(5 * num_groups, 4))  # Create subplot grid

        # Plot confusion matrix for overall data
        cm = plot_cm(y_test, y_pred_test, ax=axes[0])
        axes[0].set_title("All", fontsize=14, fontweight="bold")

        # Plot confusion matrices for each group in the dataset
        cm_dict = {"All": cm}  # Store overall confusion matrix
        for i, group in enumerate(groups):
            ax = axes[i + 1]  # Get axis for group
            subset = data_test[data_test[category] == group]  # Filter data for group
            cm = plot_cm(subset["Label"], subset["Pred"], ax=ax)  # Plot confusion matrix for group
            cm_dict[group] = cm  # Store confusion matrix for group
            ax.set_title(group, fontsize=14, fontweight="bold")

        plt.tight_layout()  # Adjust layout
        plt.show()  # Display plot
        return cm_dict  # Return dictionary of confusion matrices for each group


    def calculate_tpr(self, cms):
        """
        Calculates True Positive Rates (TPR) for each group,
        given a set of confusion matrices.
        """
        tprs = {g: cm[0, 0] / cm[0, :].sum() for g, cm in cms.items()}  # Calculate TPR
        return tprs  # Return dictionary of TPRs

############################################################## Accuracy and fairness metrics



############################################################## Bias Mitigation techniques
    
    def bias_mitigation_in_process_train(self, data: Data, config: Configuration):
        # Train the model using the sample weights calculated through reweighing
        model = RidgeClassifier(random_state=42)
        model.fit(X_train, y_train, sample_weight=sample_weights.ravel())  # Fit model with sample weights

        y_pred_test = model.predict(X_test)

        # Define the groupings for fairness analysis (Black and White) in the test set
        group_a_test = (dem_test['Ethnicity']=='Black')
        group_b_test = (dem_test['Ethnicity']=='White')

        # Get the fairness and accuracy metrics after applying reweighing
        metrics_rw = get_metrics(group_a_test, group_b_test, y_pred_test, y_test)
        display(metrics_rw)

        # Add a 'mitigation' column to both metrics dataframes to label them accordingly
        metrics_orig['mitigation'] = 'None'
        metrics_rw['mitigation'] = 'Reweighing'

        metrics = pd.concat([metrics_orig, metrics_rw], axis=0, ignore_index=True)
        display(metrics)

        # Plot the comparison of metrics between the original model and the model with reweighing
        plt.figure(figsize=(10, 6))
        sns.barplot(data=metrics, x='Metric', y='Value', hue='mitigation')
        plt.axhline(y=0.8, linewidth=2, color='r', linestyle="--")
        plt.axhline(y=-0.05, linewidth=2, color='r', linestyle="--")
        plt.axhline(y=1, linewidth=2, color='g')
        plt.axhline(y=0, linewidth=2, color='g')
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.show()

############################################################## Explainability

    def explain_model_predictions(self, blackbox_model, X_train, y_test): 
        seed = 4     
        lime = LimeTabular(blackbox_model, X_train, random_state=seed)
        show(lime.explain_local(X_test[:5], y_test[:5]), 0)

############################################################## Robustness

    def robustness_evaluation():
        import lightgbm as lgb
        import numpy as np
        from art.attacks.evasion import ZooAttack
        from art.estimators.classification import LightGBMClassifier
        from art.utils import load_mnist

        # Step 1: Load the MNIST dataset

        (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

        # Step 1a: Flatten dataset

        x_test = x_test[0:5]
        y_test = y_test[0:5]

        nb_samples_train = x_train.shape[0]
        nb_samples_test = x_test.shape[0]
        x_train = x_train.reshape((nb_samples_train, 28 * 28))
        x_test = x_test.reshape((nb_samples_test, 28 * 28))

        # Step 2: Create the model

        params = {"objective": "multiclass", "metric": "multi_logloss", "num_class": 10, "force_col_wise": True}
        train_set = lgb.Dataset(x_train, label=np.argmax(y_train, axis=1))
        test_set = lgb.Dataset(x_test, label=np.argmax(y_test, axis=1))
        model = lgb.train(params=params, train_set=train_set, num_boost_round=100, valid_sets=[test_set])

        # Step 3: Create the ART classifier

        classifier = LightGBMClassifier(model=model, clip_values=(min_pixel_value, max_pixel_value))

        # Step 4: Train the ART classifier

        # The model has already been trained in step 2

        # Step 5: Evaluate the ART classifier on benign test examples

        predictions = classifier.predict(x_test)
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
        print("Accuracy on benign test examples: {}%".format(accuracy * 100))

        # Step 6: Generate adversarial test examples
        attack = ZooAttack(
            classifier=classifier,
            confidence=0.5,
            targeted=False,
            learning_rate=1e-1,
            max_iter=200,
            binary_search_steps=100,
            initial_const=1e-1,
            abort_early=True,
            use_resize=False,
            use_importance=False,
            nb_parallel=250,
            batch_size=1,
            variable_h=0.01,
        )
        x_test_adv = attack.generate(x=x_test)

        # Step 7: Evaluate the ART classifier on adversarial test examples

        predictions = classifier.predict(x_test_adv)
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
        print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))