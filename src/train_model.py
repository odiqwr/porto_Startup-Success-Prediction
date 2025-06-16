# splitting dataset in X and Y variables
X = df.drop('status', axis = 1) 
y = df['status'] # Output/Dependent variable
# splitting dataset in train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Gradient Boosting Classifier
# model structures
gbc = GradientBoostingClassifier(learning_rate=0.02, 
      max_depth=4, random_state=100, n_estimators=1000)
gbc.fit(X_train,y_train)
# model predict
y_predicted_gb = gbc.predict(X_test)
print("Training Accuracy :", gbc.score(X_train, y_train))
print("Testing Accuracy :", gbc.score(X_test, y_test))
# model evaluation
cm = confusion_matrix(y_test, y_predicted_gb)
plt.rcParams['figure.figsize'] = (3, 3)
sns.heatmap(cm, annot = True, cmap = 'YlGnBu', fmt = '.8g')
plt.show()
# model report
cr = classification_report(y_test, y_predicted_gb)
print(cr)
print("------------------------------------------")
# calculate ROC curves
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,y_predicted_gb)
roc_auc = auc(false_positive_rate, true_positive_rate)
print("ROC Curves              =",roc_auc)
# calculate precision-recall curves
precision, recall, thresholds = precision_recall_curve(y_test, y_predicted_gb)
f1 = f1_score(y_test, y_predicted_gb)
Precision_Recall_gbs = auc(recall, precision)
print("Precision-Recall Curves =",Precision_Recall_gbs)
