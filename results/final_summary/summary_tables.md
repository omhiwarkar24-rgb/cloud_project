# Final Consolidated Results

### Security Model Comparison (Binary & Multiclass)

|Task|Model|Accuracy|Precision (Binary/Weighted)|Recall|F1-Score|ROC-AUC|FPR|FNR|Model_Path|CM_Path|
|---|---|---|---|---|---|---|---|---|---|---|
|Binary|Logistic Regression|1.0|1.0|1.0|1.0|1.0|0.0|0.0|D:\ML_CP\adaptive_cloud_project\models\security\logistic_regression_binary.joblib|D:\ML_CP\adaptive_cloud_project\results\security\plots\cm_binary_logistic_regression.png|
|Binary|Decision Tree|1.0|1.0|1.0|1.0|1.0|0.0|0.0|D:\ML_CP\adaptive_cloud_project\models\security\decision_tree_binary.joblib|D:\ML_CP\adaptive_cloud_project\results\security\plots\cm_binary_decision_tree.png|
|Binary|Random Forest|1.0|1.0|1.0|1.0|1.0|0.0|0.0|D:\ML_CP\adaptive_cloud_project\models\security\random_forest_binary.joblib|D:\ML_CP\adaptive_cloud_project\results\security\plots\cm_binary_random_forest.png|
|Binary|XGBoost|0.998|0.9957356076759062|1.0|0.997863247863248|1.0|0.0037523452157598|0.0|D:\ML_CP\adaptive_cloud_project\models\security\xgboost_binary.joblib|D:\ML_CP\adaptive_cloud_project\results\security\plots\cm_binary_xgboost.png|
|Binary|MLP Neural Network|1.0|1.0|1.0|1.0|1.0|0.0|0.0|D:\ML_CP\adaptive_cloud_project\models\security\mlp_neural_network_binary.joblib|D:\ML_CP\adaptive_cloud_project\results\security\plots\cm_binary_mlp_neural_network.png|
|Multiclass|Logistic Regression|1.0|1.0|1.0|1.0|1.0|0.0|0.0|D:\ML_CP\adaptive_cloud_project\models\security\logistic_regression_multiclass.joblib|D:\ML_CP\adaptive_cloud_project\results\security\plots\cm_multiclass_logistic_regression.png|
|Multiclass|Decision Tree|1.0|1.0|1.0|1.0|1.0|0.0|0.0|D:\ML_CP\adaptive_cloud_project\models\security\decision_tree_multiclass.joblib|D:\ML_CP\adaptive_cloud_project\results\security\plots\cm_multiclass_decision_tree.png|
|Multiclass|Random Forest|1.0|1.0|1.0|1.0|1.0|0.0|0.0|D:\ML_CP\adaptive_cloud_project\models\security\random_forest_multiclass.joblib|D:\ML_CP\adaptive_cloud_project\results\security\plots\cm_multiclass_random_forest.png|
|Multiclass|XGBoost|1.0|1.0|1.0|1.0|1.0|0.0|0.0|D:\ML_CP\adaptive_cloud_project\models\security\xgboost_multiclass.joblib|D:\ML_CP\adaptive_cloud_project\results\security\plots\cm_multiclass_xgboost.png|
|Multiclass|MLP Neural Network|0.999|0.999001872659176|0.999|0.9989843395713034|1.0|0.0001428571428571|0.001|D:\ML_CP\adaptive_cloud_project\models\security\mlp_neural_network_multiclass.joblib|D:\ML_CP\adaptive_cloud_project\results\security\plots\cm_multiclass_mlp_neural_network.png|

### Predictive Maintenance Model Comparison

|Model|Accuracy|Precision|Recall|F1-Score|ROC-AUC|Model_Path|CM_Path|
|---|---|---|---|---|---|---|---|
|Logistic Regression|0.9|0.75|0.9617224880382776|0.8427672955974843|0.9590515525917802|D:\ML_CP\adaptive_cloud_project\models\maintenance\logistic_regression_maintenance.joblib|D:\ML_CP\adaptive_cloud_project\results\maintenance\plots\cm_logistic_regression.png|
|Decision Tree|0.8986666666666666|0.7587548638132295|0.9330143540669856|0.8369098712446352|0.9282738858573084|D:\ML_CP\adaptive_cloud_project\models\maintenance\decision_tree_maintenance.joblib|D:\ML_CP\adaptive_cloud_project\results\maintenance\plots\cm_decision_tree.png|
|Random Forest|0.9133333333333332|0.7666666666666667|0.9904306220095692|0.8643006263048016|0.9533116946289434|D:\ML_CP\adaptive_cloud_project\models\maintenance\random_forest_maintenance.joblib|D:\ML_CP\adaptive_cloud_project\results\maintenance\plots\cm_random_forest.png|
|XGBoost|0.9053333333333332|0.7593984962406015|0.9665071770334928|0.8505263157894737|0.9581317602525892|D:\ML_CP\adaptive_cloud_project\models\maintenance\xgboost_maintenance.joblib|D:\ML_CP\adaptive_cloud_project\results\maintenance\plots\cm_xgboost.png|
|MLP Neural Network|0.896|0.7811158798283262|0.8708133971291866|0.8235294117647058|0.959060396748888|D:\ML_CP\adaptive_cloud_project\models\maintenance\mlp_neural_network_maintenance.joblib|D:\ML_CP\adaptive_cloud_project\results\maintenance\plots\cm_mlp_neural_network.png|

### Adaptive Learning Adaptation Metrics

|Model_Version|Accuracy|Precision|Recall|F1-Score|ROC-AUC|
|---|---|---|---|---|---|
|Old Model (Pre-Drift)|0.75|0.75|1.0|0.8571428571428571|0.7859555555555555|
|New Adaptive Model|1.0|1.0|1.0|1.0|1.0|

