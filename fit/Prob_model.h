#ifndef PROB_MODEL_H_INCLUDED
#define PROB_MODEL_H_INCLUDED

#include <vector>
#include <fstream>
#include <cmath>
#include <cassert>
#include <cstdlib>
#include "Point.h"
using namespace std;


//--------------------------------------------------------------


class Prob_model { 
public:
  //c are the parameters(coefficients) in the model
  //p are the data points
  //returns log(P(D|M))
  virtual double probability(const vector <double> & c)=0;
  virtual void gradient(const vector <double> & c, 
                        vector<double> & grad);
  virtual double gradient(const vector <double> & c, 
                        int dir);
  
  virtual void read(istream & is)=0;
  virtual void generate_guess(vector<double> & c)=0;
  virtual bool is_ok(const vector <double> & c) { return true;} ;
  
  virtual void niceprint(const vector <double> & c, const vector <double> & cerr)=0;
  virtual ~Prob_model() { }
};


//--------------------------------------------------------------

class Morse_model:public Prob_model { 
public:
  virtual double probability(const vector <double> & c);
  virtual void gradient(const vector <double> & c, 
                        vector<double> & grad);
  virtual double gradient(const vector <double> & c, 
                          int dir);
  virtual void read(istream & is);
  virtual bool is_ok(const vector <double> & c);
  virtual void niceprint(const vector <double> & c, const vector <double> & cerr);
  virtual void generate_guess(vector<double> & c);

private:
  vector <Point> data;
};


/*
 c[0]*x+c[1]
 */
class Linear_model:public Prob_model { 
public:
  virtual double probability(const vector <double> & c);
  virtual void read(istream & is);
  virtual void niceprint(const vector <double> & c, const vector <double> & cerr);
  virtual void generate_guess(vector<double> & c);
  
private:
    vector <Point> data;
};



/*
 c[0]+c[1]*(x-c[2])**2
 */
class Quadratic_model:public Prob_model { 
public:
  virtual double probability(const vector <double> & c);
  virtual void read(istream & is);
  virtual void niceprint(const vector <double> & c, const vector <double> & cerr);
  virtual void generate_guess(vector<double> & c);
  
private:
    vector <Point> data;
};

/*
 c[0]+c[1]*(x-c[2])**2
 */
class Cubic_model:public Prob_model { 
public:
  virtual double probability(const vector <double> & c);
  virtual void read(istream & is);
  virtual void niceprint(const vector <double> & c, const vector <double> & cerr);
  virtual void generate_guess(vector<double> & c);
  
private:
    vector <Point> data;
};

/*
 * Quartic potential assuming that x=0 is a symmetry point.
 * c[0]+c[1]*x^2+c[2]*x^4
 * */

class Centered_quartic:public Prob_model { 
public:
  virtual double probability(const vector <double> & c);
  virtual void read(istream & is);
  virtual void niceprint(const vector <double> & c, const vector <double> & cerr);
  virtual void generate_guess(vector<double> & c);
  
private:
    vector <Point> data;
};


/*!
 * fit a number of lines to common linear functions.  The data format is as follows:
 * l x1 x2 ...  e err
 *
 * The model is then 
 * E(x1,x2,...)= sum c_i x_i  + sum E_l
 * This was written originially to fit Lennard-Jones parameters to multiple 
 * potential energy surfaces obtained by moving different monolayer configurations 
 * of water away from a graphene surface
 * */
class Multiple_linear:public Prob_model { 
public:
  virtual double probability(const vector <double> & c);
  virtual void read(istream & is);
  virtual void niceprint(const vector <double> & c, const vector <double> & cerr);
  virtual void generate_guess(vector<double> & c);
  
private:
    vector <Point> data;
    int nlines;
    int ndim;
};

//--------------------------------------------------------------

//###########################################################################


class Prob_distribution { 
public:
  Prob_distribution(int np, double mi,
                    double spac) {
    npoints=np;
    mymin=mi;
    spacing=spac;
    density.resize(npoints);
    for(vector<double>::iterator i=density.begin();
        i!=density.end(); i++) *i=0;
  }
  //-----------------------------------------------------
  
  void accumulate(double a, double prob) {
    int place=int((a-mymin+spacing/2)/spacing);
    //cout << a <<" place " << place << endl;
    if(place >0 && place<npoints){
      density[place]+=prob; 
      //cout << "prob " << prob << endl;
    }
  }
  
  //-----------------------------------------------------

  void write(const string name) { 
    ofstream out(name.c_str());
    double sum=0;
    for(vector<double>::iterator i=density.begin();
        i!=density.end(); i++) sum+=*i;
    
    for(int i=0; i< npoints; i++) {
      out << mymin+spacing*i << "   " << density[i]/sum << endl;
    }
  }
  
  
  
  //-----------------------------------------------------
  void avg(double & avg, double & err);
  //-----------------------------------------------------

  double skew();
  
  void clear() { 
    for(vector<double>::iterator i=density.begin();
        i!= density.end(); i++) *i=0.0;
  }
  
private:
  vector <double> density;
  double mymin;
  double spacing;
  int npoints;
};

//###########################################################################

int check_model_consistency(Prob_model & mod, vector <double> & c);


//c should contain the best guess..will be returned as the maximum probability point
//data should be data on a line..
//min will be the average min 
void sample_min(vector <double> & c, Prob_model & mod); 
//------------------------------------------------------------------


#endif //PROB_MODEL_H_INCLUDED
