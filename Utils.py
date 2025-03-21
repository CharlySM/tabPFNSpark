from tabPFNSpark import predictions, count

from tabpfn import TabPFNClassifier
import pandas as pd


def applyTabPFN(rows, columns, test_df):
    """
    @briefs Apply tabPFN to partition in rows
    @details This function apply tabPFN to partitions content in rows, then the function do the prediction,
     the columns used into train and prediction are in param columns, the test used is in test_df param.
     The predictions are added to prediction variable to the rest of partitions predictions.
    @param rows Partition to train .
    @param columns Columns used to trained.
    @param test_df Dataset to generate predictions.
    """
    pandas_partition = pd.DataFrame(rows, columns=columns)
    X_train = pandas_partition.drop("label", axis=1)  # Reemplaza "target_column"
    y_train = pandas_partition["label"]

    test_df = test_df[test_df["partition_id"] == count]
    count += 1

    # Use test_df for evaluation
    columnsTest = [i for i in columns if i not in ["label", "partition_id"]]
    X_val = test_df[columnsTest] # Drop partition_id before prediction
    y_val = test_df["label"]

    model = TabPFNClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val) # Predict on the test_df
    predictions.append(y_pred)

# Función de entrenamiento para cada partición.
def train_and_evaluate(partition, columns, test_df): # Add test_df as an argument
    """
    @briefs It is trained the dataset partition with tabPFN and predict using test_df variable
    @details This function use tabPFN to train dataset partition and predict the result using dataset test_df,
    the columns used for the training are contained in columns param then is returned the predictions for partitions.
    @param rows Partition to train .
    @param columns Columns used to trained.
    @param test_df Dataset to generate predictions.
    @return It is returned the predictions of partitions to the rest of predictions in dataset.
    """
    rows = list(partition)
    if not rows:  # Check if partition is empty
        return None

    applyTabPFN(rows, columns, test_df)

    # Create a list of dictionaries with the required structure
    return predictions