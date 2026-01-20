import matplotlib.pyplot as plt
import os


def actual_vs_predicted(y_test, y_pred, save_path):
    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        linestyle="--"
    )
    plt.xlabel("Actual Selling Price")
    plt.ylabel("Predicted Selling Price")
    plt.title("Actual vs Predicted Car Price")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "actual_vs_predicted.png"))
    plt.show()


def residual_plot(y_test, y_pred, save_path):
    residuals = y_test - y_pred

    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted Price")
    plt.ylabel("Residual Error")
    plt.title("Residual Error Analysis")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "residual_plot.png"))
    plt.show()


def feature_importance_plot(model, feature_names, save_path):
    importance = model.feature_importances_

    plt.figure(figsize=(7, 5))
    plt.barh(feature_names, importance)
    plt.xlabel("Importance Score")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "feature_importance.png"))
    plt.show()
