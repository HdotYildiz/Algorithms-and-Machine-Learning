import numpy as np


class LinearRegression:
    """
    Multi variable linear regression
    - Includes closed form and gradient descent weight training
    - Includes L1, L2 and elastic net regularization
    """

    def __init__(self, a: float = 0.00003) -> None:
        self.w = np.array([0])
        self.a = a

    def close_form_fit(self, x: np.array, y: np.array, alpha: float = 0) -> None:
        """
        w = (X^T X + aI)^-1 X^T y

        Parameters:
        alpha: influence rate of the Ridge factor for regularization
        """
        x = self.add_bias_column(x)

        ridge = np.array([0])
        if alpha:
            ridge = np.identity(x.shape[1])
            ridge[0][0] = 0

        self.w = np.linalg.inv(x.T.dot(x) + alpha * ridge).dot(x.T).dot(y)

    def gradient_descent(
        self, x: np.array, y: np.array, regularize: str = "", alpha: float = 0, l1_ratio: float = 0.5
    ) -> None:
        """
        dw = 2/n (w^T X - Y) X^T + a(regularization factor)
        """
        reg_factor = 0
        if regularize == "L1":
            reg_factor = alpha * np.sum(np.abs(self.w[1:]))
        elif regularize == "L2":
            reg_factor = alpha * np.sum(np.square(self.w[1:]))
        elif regularize == "elastic":
            reg_factor = alpha * l1_ratio * np.sum(np.abs(self.w[1:])) + (1 - l1_ratio) / 2 * alpha * np.sum(
                np.square(self.w[1:])
            )

        n = y.shape[0]
        dr_dw = 2 / n * x.T.dot(x.dot(self.w) - y) + reg_factor

        self.w = self.w - self.a * dr_dw

    def descent(
        self,
        x: np.array,
        y: np.array,
        regularize: str = "",
        alpha: float = 0,
        epochs: float = 2e6,
        eval_per_epoch: float = 1e5,
        verbose: bool = False,
    ) -> None:
        x = self.add_bias_column(x)

        self.w = np.array([0] * x.shape[1])
        prior_loss = 1e32
        for e in range(int(epochs)):
            self.gradient_descent(x, y, regularize=regularize, alpha=alpha)

            if e % eval_per_epoch == 0:
                loss = self.average_loss(x, y)
                if verbose:
                    print(f"Epoch {e}, w = {self.w} has loss: {self.average_loss(x, y)}")

                if loss < prior_loss * (1 - 1e-8):
                    prior_loss = loss
                else:
                    break

    def average_loss(self, x: np.array, y: np.array) -> float:
        loss = (y - x.dot(self.w)) ** 2
        return np.mean(loss)

    def add_bias_column(self, x: np.array) -> np.array:
        xb = np.array([np.ones(x.shape[0])]).T
        return np.append(xb, x, axis=1)

    def fit(self, x: np.array, y: np.array, regularize: str = "", alpha: float = 0, **kwargs) -> None:
        """
        Uses the closed loop equation if there are less than 100k features, otherwise does gradient descent.

        Parameters:
        regularize: selection of regularization methods; L1, L2, elastic
        alpha: alpha factor of regularized factor
        """
        if x.shape[1] < 1e5 and (not regularize or regularize == "L2"):
            self.close_form_fit(x, y, alpha)
        else:
            self.descent(x, y, regularize, alpha, **kwargs)

    def predict(self, x: np.array) -> float:
        x = self.add_bias_column(x)
        return self.w * x


if __name__ == "__main__":
    import pandas as pd
    from sklearn import linear_model

    df = pd.read_csv("./data/Advertising.csv", index_col=0)

    x = np.array(df[["TV", "radio", "newspaper"]].values)
    y = np.array(df["sales"].values)

    lr = LinearRegression()
    lr.fit(x, y)
    print(f"Custom LR weights: {lr.w}")

    sk_lr = linear_model.LinearRegression().fit(x, y)
    print(f"Sklearn LR weights: {np.append(sk_lr.intercept_, sk_lr.coef_)}")

    lr.fit(x, y, regularize="L1", alpha=1)
    print(f"Custom LassoR weights: {lr.w}")

    lassor = linear_model.Lasso(alpha=1)
    lassor.fit(x, y)
    print(f"Sklearn LassoR weights: {np.append(lassor.intercept_, lassor.coef_)}")

    lr.fit(x, y, regularize="L2", alpha=1)
    print(f"Custom RR weights: {lr.w}")

    rr = linear_model.Ridge(alpha=1, solver="cholesky")
    rr.fit(x, y)
    print(f"Sklearn RR weights: {np.append(rr.intercept_, rr.coef_)}")
