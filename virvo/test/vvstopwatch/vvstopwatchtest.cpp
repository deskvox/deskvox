#include <iostream>

#include "vvclock.h"

using namespace std;

int main(int, char**)
{
  vvStopwatch* watch;
  char input[128];

  watch = new vvStopwatch();

  cerr << "Current time in seconds: " << watch->getTime() << endl;
  cerr << "Input something to start stopwatch: " << endl;
  cin >> input;
  watch->start();

  cerr << "Input something to get time since start: " << endl;
  cin >> input;
  cerr << "Time since start: " << watch->getTime() << endl;

  cerr << "Input something to get Time since last call: " << endl;
  cin >> input;
  cerr << "Time difference: " << watch->getDiff() << endl;

  cerr << "Input something to get total time: " << endl;
  cin >> input;
  cerr << "Total time: " << watch->getTime() << endl;

  delete watch;

  return 0;
}
