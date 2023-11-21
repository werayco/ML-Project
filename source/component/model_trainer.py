# best_model_score=[model_name for model_name,score in model_report.items() if score == best_model_score]

def evaluate_model(x_train,y_train,x_test,y_test,models):
    try:
        report ={}

        for model in models.value():
            model.fit(x_train,y_train)
            y_test_pred=model.predict(x_train)
            y_test_pred=model.predict(x_test)
            train_model_score=r2_score(y_train, y_train_pred)
            test_model_score=r2_score(y_test,y_test_pred)
            report[list(models.keys())]=test_model_score
            return report