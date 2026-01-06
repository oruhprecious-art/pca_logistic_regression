{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "b7491ad2-97e2-4eb0-bb38-c7d24c1cdc10",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Explained Variance Ratio: [0.44272026 0.18971182]\n"
     ]
    }
   ],
   "source": [
    "# Import required libraries\n",
    "\n",
    "from sklearn.datasets import load_breast_cancer\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.decomposition import PCA\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import accuracy_score\n",
    "import pandas as pd\n",
    "\n",
    "# Load Cancer data set: This dataset contains tumor measurements and labels (malignant or benign).\n",
    "\n",
    "cancer = load_breast_cancer()\n",
    "X = cancer.data     # features\n",
    "y = cancer.target   # labels (0 = malignant, 1 = benign)\n",
    "\n",
    "# Standardize the data\n",
    "scaler = StandardScaler()\n",
    "X_scaled = scaler.fit_transform(X)\n",
    "\n",
    "# Apply PCA (Reduce to 2 Components)\n",
    "pca = PCA(n_components=2)\n",
    "X_pca = pca.fit_transform(X_scaled)\n",
    "print(\"Explained Variance Ratio:\", pca.explained_variance_ratio_)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "30422433-7a0d-483d-a368-f5e20b1ee83b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "       PCA1       PCA2  Target\n",
      "0  9.192837   1.948583       0\n",
      "1  2.387802  -3.768172       0\n",
      "2  5.733896  -1.075174       0\n",
      "3  7.122953  10.275589       0\n",
      "4  3.935302  -1.948072       0\n"
     ]
    }
   ],
   "source": [
    "# Convert PCA Output to a DataFrame\n",
    "\n",
    "pca_df = pd.DataFrame(\n",
    "    data=X_pca,\n",
    "    columns=[\"PCA1\", \"PCA2\"]\n",
    ")\n",
    "\n",
    "pca_df[\"Target\"] = y\n",
    "print(pca_df.head())\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "68bb1c60-ec8f-44c4-8c7d-c87048e16a72",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Logistic Regression Accuracy: 0.9912280701754386\n"
     ]
    }
   ],
   "source": [
    "# Logistic Regression for Prediction\n",
    "\n",
    "# Split data\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X_pca, y, test_size=0.2, random_state=42\n",
    ")\n",
    "\n",
    "# Train the model\n",
    "model = LogisticRegression(max_iter=1000)\n",
    "model.fit(X_train, y_train)\n",
    "\n",
    "# Make predictions & evaluate\n",
    "y_pred = model.predict(X_test)\n",
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "\n",
    "print(\"Logistic Regression Accuracy:\", accuracy)\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c3286f8a-b59c-41de-97a1-9af63ba0cf5f",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
