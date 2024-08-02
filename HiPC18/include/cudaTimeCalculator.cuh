#pragma once

class cudaTimeCalculator {
 public:
  cudaTimeCalculator();
  ~cudaTimeCalculator();

  void startClock();
  void endClock();
  float getTime();

 private:
  cudaEvent_t _star;
  cudaEvent_t _stop;

  float _time;
};

