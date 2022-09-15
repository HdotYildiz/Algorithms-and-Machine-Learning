import matplotlib.pyplot as plt


class LinearRegressionSingle:
    """
    Single variable linear regression
    """

    def __init__(self, a=0.001):
        self.w = 0
        self.b = 0
        self.a = a

    def gradient_descent(self, x, y):
        dr_dw = 0
        dr_db = 0
        n = len(y)

        for i in range(n):
            dr_dw += -2 * x[i] * (y[i] - (self.w * x[i] + self.b))
            dr_db += -2 * (y[i] - (self.w * x[i] + self.b))

        self.w = self.w - (dr_dw / float(n)) * self.a
        self.b = self.b - (dr_db / float(n)) * self.a

    def average_loss(self, x, y):
        n = len(y)
        loss = 0

        for i in range(n):
            loss += (y[i] - (self.w * x[i] + self.b)) ** 2

        return loss / n

    def fit(self, x, y, epochs=400, print_per_epoch=10, plot=False):
        for e in range(epochs):
            self.gradient_descent(x, y)

            if e % print_per_epoch == 0:
                print(f"Epoch {e}, w = {round(self.w, 5)}, b = {round(self.b, 5)} has loss: {self.average_loss(x, y)}")

                if plot:
                    self.plot(x, y)

    def predict(self, x):
        return self.w * x + self.b

    def plot(self, x, y):
        plt.plot(x, [self.predict(value) for value in x], color="#58b970", label="Regression Line")
        plt.scatter(x, y, c="#ef5423", label="data points")

        plt.legend()
        plt.show()
