#ifndef MIN_H_INCLUDED
#define MIN_H_INCLUDED

#include "macopt.h"
#include "Point.h"
#include "Prob_model.h"
#include <vector>
#include <iostream>
using namespace std;




//------------------------------------------------------------------------

class Least_squares_opt:public Macopt { 
public:
  Least_squares_opt(int _n,int _verbose, double _tol,
                    int _itmax,int _rich):Macopt(_n,_verbose,_tol,_itmax,_rich) {
                    } 
  double func(double * _p) { 
    
    double f=0;
    for(int i=0; i< c.size(); i++) c[i]=_p[i+1];
    f=-mod->probability(c);
    return f;
     
  }
  
  double dfunc(double * _p, double * _g) {
    int m=c.size()+1;
    double base=func(_p);
    double step=1e-8;
    for(int i=1; i< m; i++) {
      _p[i]+=step;
      double nwfunc=func(_p);
      _p[i]-=step;
      _g[i]=(nwfunc-base)/step;
    }
    _g[0]=base;
    return base;
  }
  void iteration_print(double f, double gg, double tol,  int itn) {
    //cout <<"macopt: ";
    //for(int i=0; i< c.size(); i++) cout << c[i] << " ";
    //cout <<  endl;
  }
  
  void optimize() { 
    double p[c.size()+1];
    for(int i=0; i< c.size(); i++) p[i+1]=c[i];
    //maccheckgrad(p,c.size(),.0001, c.size());
    macoptII(p,c.size());
  }
  
  Prob_model * mod;
  vector <double>  c; 
  
};


//------------------------------------------------------------------


#endif //MIN_H_INCLUDED

