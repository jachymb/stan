parameters {
  real x;
  real y;
}
model {
  print("log density evaluated");
  target += -(pow(1 - x, 2) + 100 * pow(y - pow(x, 2), 2));
}
