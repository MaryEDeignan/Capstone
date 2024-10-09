import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as torchfunct

class SCADLinearRegression(nn.Module):
  def __init__(self, input_size, lambda_val, a_val, random_state=440):
    super(SCADLinearRegression, self).__init__()
    self.input_size = input_size
    self.lambda_val = lambda_val
    self.a_val = a_val

    self.linear = nn.Linear(input_size, 1, bias = True, dtype=torch.float64, device = torch.device('cpu'))


  def scad_penalty(self, beta_hat):
    """Function calculated SCAD penalty based on the weights (beta_hat) and determines which weights fall into the linear, quadratic,
       or constant penalty zones. Then calculates the linear, quadratic, and constant parts of the penalty. The function is
       based on code in Andy Jones 'The smoothly clipped absolute deviation (SCAD) penalty' Github."""
    is_linear = (torch.abs(beta_hat) <= self.lambda_val).type(torch.float64)
    is_quadratic = torch.logical_and(self.lambda_val < torch.abs(beta_hat), torch.abs(beta_hat) <= self.a_val * self.lambda_val).type(torch.float64)
    is_constant = (self.a_val * self.lambda_val) < torch.abs(beta_hat)

    linear_part = self.lambda_val * torch.abs(beta_hat) * is_linear
    quadratic_part = (2*self.a_val * self.lambda_val * torch.abs(beta_hat) - beta_hat**2 - self.lambda_val**2)/(2 * (self.a_val - 1)) * is_quadratic
    constant_part = (self.lambda_val**2 * (self.a_val + 1)) / 2 * is_constant

    return linear_part + quadratic_part + constant_part # Return the total SCAD penalty

  def scad_derivative(self,beta_hat): # from Andy Jones Github Repository
    """Function calculates the derivative of the SCAD penalty with respect to the weights. The function is based on code
       in Andy Jones 'The smoothly clipped absolute deviation (SCAD) penalty' Github."""
    return self.lambda_val * ((beta_hat <= self.lambda_val).type(torch.float64) +
                              (self.a_val * self.lambda_val - beta_hat) * ((self.a_val * self.lambda_val - beta_hat) > 0).type(torch.float64) /
                             ((self.a_val - 1) * self.lambda_val) * (beta_hat > self.lambda_val).type(torch.float64))

  def forward(self, X):
    """Forward pass of the model."""
    return self.linear(X)

  def loss(self, X, y):
    """Function calculates the total loss for the model including both MSE and SCAD penalty."""
    y_pred = self.forward(X)

    if y.dim() == 1:  # If y is only one dimension, reshape to 2 dimensions
        y = y.unsqueeze(1)
    elif y.dim() > 2:  # If y is too big, remove extra dimensions
        y = y.squeeze()

    mse = nn.MSELoss()(y_pred, y)
    penalty = torch.sum(self.scad_penalty(self.linear.weight))

    return mse + penalty


  def fit(self, x, y, num_epochs= 2000, learning_rate= .001):
    """Trains the SCAD model using stochastic gradient descent."""
    optimize = optim.SGD(self.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
      self.train()
      optimize.zero_grad() 
      y_pred = self(x)
      loss_with_scad = self.loss(x, y)
      loss_with_scad.backward() 
      optimize.step()

      if (epoch+1) % 100 == 0:
        print(f'epoch: {epoch+1}/{num_epochs}, loss_with_scad: {loss_with_scad.item()}')
    return
                

  def predict(self, X):
    """Predicts the target values using the trained SCAD model."""
    self.eval()
    with torch.no_grad():
      y_pred = self(X)
    return y_pred

  def get_coefficients(self):  
    """Returns the learned coefficients from the linear regression model."""
    return self.linear.weight