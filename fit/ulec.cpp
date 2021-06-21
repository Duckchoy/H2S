//----------------------------------------------------------------------
//src/ulec.cpp

#include "ulec.h"

Random_generator rng;


double unif()
{
  static int ix=1234567;
  int k1=ix/127773;
  ix=16807*(ix-k1*127773)-k1*2836;
  if(ix < 0)
    ix+=2147483647;
  return 4.656612875e-10 * ix;
}


double ranr2exponential() {
  while(1) { 
    double ex=-log(rng.ulec())/.4;
    //cout << "ex " << ex  << "  prob " << (.5/.8)*ex*ex*exp(-ex+0.4*ex) << endl;
    if(rng.ulec()< (.5/.8)*ex*ex*exp(-ex+0.4*ex) ) { 
      //cout << "acc " << endl;
       return ex;
    }
  }
}

double rancos() {
  return acos(1-2*rng.ulec());
}

//----------------------------------------------------------------------


//----------------------------------------------------------------------
