#ifndef POINT_H_INCLUDED
#define POINT_H_INCLUDED
#include <vector>
#include <fstream>
using namespace std;
//------------------------------------------------------------------

class Point { 
public:
  vector <double> x;
  double en;
  double err;
  int age;
  Point() { 
    age=0;
  }
  void write(ostream & os) { 
    os << "pos ";
    for(vector<double>::iterator i=x.begin(); i!=x.end();
        i++) os << *i << "  ";
    os << " en " << en << " err " << err<< endl;
  }
};



//------------------------------------------------------------------
//probability that f(x1) > f(x2)
inline double prob_greater(const Point & pt1, const Point & pt2) { 
  double prob=1.0;
  double er=sqrt(pt1.err*pt1.err+pt2.err*pt2.err);
  double dif=pt2.en-pt1.en;
  return erfc(dif/(sqrt(2)*er))/2.0;
}


//dir must be a unit vector..
inline void translate(const vector <double> & x, vector <double> & xp,
               const vector <double> & dir, const double dist)  { 
  assert(x.size()==xp.size());
  assert(x.size()==dir.size());
  int ndim=x.size();
  for(int i=0; i< ndim; i++) { 
    xp[i]=dist*dir[i]+x[i];
  }
}

#endif