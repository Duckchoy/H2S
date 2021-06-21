
#include "Prob_model.h"
#include <iostream>
#include "ulec.h"
#include <cassert>
#include <string>

typedef vector<double>::iterator dit_t;


void split(string & text, string & separators, vector<string> & words) {
  int n = text.length();
  int start, stop;
  
  start = text.find_first_not_of(separators);
  while ((start >= 0) && (start < n)) {
    stop = text.find_first_of(separators, start);
    if ((stop < 0) || (stop > n)) stop = n;
    words.push_back(text.substr(start, stop - start));
    start = text.find_first_not_of(separators, stop+1);
  }
}

//###########################################################################

//---------------------------------------------------------------------

void read_into_data(istream & is, vector <Point> & data) { 
  Point pt;
  string line; string space=" ";
  while(getline(cin, line)) { 
    vector <string> words;
    split(line, space, words);
    if(words.size() > 0) { 
      int ndim=words.size()-2;
      pt.x.resize(ndim);
      for(int d=0; d< ndim; d++) {
        pt.x[d]=atof(words[d].c_str());
      }
      pt.en=atof(words[ndim].c_str());
      pt.err=atof(words[ndim+1].c_str());
      //if(pt.x[0] > 0.81 && pt.x[0] < 1.11
      //   && pt.x[1] > 1.01 && pt.x[1] < 2.39) 
      data.push_back(pt);
    }
  }
}


void Prob_model::gradient(const vector <double> & c,
                           vector <double> & grad) { 
  double delta=1e-12;
  int n=c.size();
  grad.resize(n);
  vector <double> cp;
  double base=probability(c);
  for(int i=0; i< n; i++) { 
    cp=c;
    cp[i]+=delta;
    double px=probability(cp);
    grad[i]=(px-base)/delta;
  }
}
//---------------------------------------------------------------------

double Prob_model::gradient(const vector <double> & c,
                             int dir) { 
  double delta=1e-12;
  int n=c.size();
  vector <double> cp;
  double base=probability(c);
  cp=c;
  cp[dir]+=delta;
  double px=probability(cp);
  return (px-base)/delta;
}

//---------------------------------------------------------------------

void Morse_model::read(istream & is) { 
  read_into_data(is, data);
}
//------------------------------------------------------------------------

void Morse_model::generate_guess(vector <double> & c) { 
  int ndim=data[0].x.size();
  int nparms=1+2*ndim+ndim*(ndim+1)/2;
  c.resize(nparms);
  //cout << " ndim " << ndim << " nparms " << nparms << endl;
  
  vector <double> maxd(ndim), mind(ndim);
  for(int d=0; d< ndim; d++) { 
    maxd[d]=-1e99; mind[d]=1e99;
    for(vector<Point>::iterator i=data.begin(); 
        i!= data.end(); i++) { 
      if(maxd[d] < i->x[d]) maxd[d]=i->x[d];
      if(mind[d] > i->x[d]) mind[d]=i->x[d];
    }
  }

  for(dit_t i=c.begin(); i!=c.end(); i++) *i=4*rng.ulec();
  c[0]=data[0].en;
  for(int d=0; d< ndim; d++) { 
    c[2*d+1]=mind[d]+(maxd[d]-mind[d])*(0.5+rng.gasdev()*.1);
  }
  
}

//------------------------------------------------------------------------
//does the matrix multiplication for a general Morse function
inline double fval_quad(const vector <double> & c,const vector <double> & x) { 
  double f=c[0];
  int n=x.size();
  assert(c.size() >= 1+2*n+n*(n+1)/2);
  int count=1;
  vector <double> xshift;
  for(int i=0; i< n; i++) { 
    xshift.push_back(1-exp(-c[count+1]*(x[i]-c[count])));
    //xshift.push_back(x[i]-c[count]);
    count++; count++;
  }
  
  for(int i=0; i< n; i++) { 
    for(int j=i; j< n; j++) { 
      double fac=1.0;
      if(i!=j) fac=2.0;
      f+=fac*xshift[i]*c[count]*xshift[j];
      count++;
    }
  }
  return f;
}

//---------------------------------------------------------------------

double Morse_model::probability(const vector <double> & c) { 

  double f=0;
  
  int ndatapts=data.size();
  for(vector<Point>::const_iterator i=data.begin(); i != data.end();
      i++) { 
    double fp=fval_quad(c,i->x);
    f-=(i->en-fp)*(i->en-fp)/(2*i->err*i->err);
  }
  return f;
}

//---------------------------------------------------------------------

void Morse_model::gradient(const vector <double> & c,
                               vector <double> & grad) { 
  double delta=1e-12;
  int n=c.size();
  grad.resize(n);
  vector <double> cp;
  double base=probability(c);
  for(int i=0; i< n; i++) { 
    cp=c;
    cp[i]+=delta;
    double px=probability(cp);
    grad[i]=(px-base)/delta;
  }
}
//---------------------------------------------------------------------

double Morse_model::gradient(const vector <double> & c,
                               int dir) { 
  double delta=1e-9;
  int n=c.size();
  vector <double> cp;
  double base=probability(c);
  cp=c;
  cp[dir]+=delta;
  double px=probability(cp);
  return (px-base)/delta;
}

//---------------------------------------------------------------------
bool Morse_model::is_ok(const vector <double> & c) {
  int count=1;
  int n=data[0].x.size();
  
  for(int i=0; i< n; i++) { 
    double min=1e99, max=-1e99;
    for(vector<Point>::const_iterator j=data.begin(); j!= data.end(); j++) { 
      if(j->x[i] < min) min=j->x[i];
      if(j->x[i] > max) max=j->x[i];
    }
    //if(c[count] < 0) { return false; }
    if (c[count] < min || c[count] > max) {  
      cout << "rejecting an maximal point because it's out of range of the data set.  You may want"
      " to check if you're really bracketing the minimum." 
      << " direction " << i << " value " << c[count] << endl;
      return false; 
    }
    //if(c[count+1] < 0) {  return false; }
    count+=2;
  }
  
  for(int i=0; i< n; i++) { 
    for(int j=i; j< n; j++) { 
      if(i==j && c[count] < 0) {  
        cout << "rejection because of negative correlation matrix \n";
        return false;}
      count++;
    }
  }
  return true;
  
}

//---------------------------------------------------------------------
void Morse_model::niceprint(const vector <double> & c, const vector <double> & cerr) {
  int n=data[0].x.size();
  
  cout << "min energy " << c[0] << " +/- " << cerr[0] << endl;
  assert(c.size() >= 1+2*n+n*(n+1)/2);
  int count=1;
  vector <double> xshift;
  for(int i=0; i< n; i++) { 
    cout << "gmin " << c[count] << " +/- " << cerr[count] << " b " << c[count+1] 
    << " +/- " << cerr[count+1] << endl;
    count++; count++;
  }
  
  cout << "correlation matrix " << endl;
  for(int i=0; i< n; i++) { 
    int counta=count;
    for(int j=i; j< n; j++) { 
      cout << c[count] << " ";
      count++;
    }
    
    cout << " +/- ";
    for(int j=i; j< n; j++) { 
      cout << cerr[counta] << " ";
      counta++;
    }
    cout << endl;
  }
  
  if(n==1) { 
    cout << "gplot function: f(x)="
    << c[0]<< " + " << c[3] << "*(1-exp(-"<< c[2] 
    << "*(x-" << c[1] << ")))**2\n";
  }
  
}


//###########################################################################
//---------------------------------------------------------------------

void Linear_model::read(istream & is) { 
  read_into_data(is, data);
}
//------------------------------------------------------------------------

void Linear_model::generate_guess(vector <double> & c) { 
  int ndim=data[0].x.size();
  assert(ndim==1);
  int nparms=2;
  c.resize(nparms);
  //cout << " ndim " << ndim << " nparms " << nparms << endl;
  //this is probably not terribly optimal, but it shouldn't be too hard to brute-force our way through it.
  for(int d=0; d< 1; d++) { 
    c[d]=30*(0.5+rng.gasdev()*.1);
  }
  c[1]=data[0].x[0]+0.5*rng.gasdev();
  
}

//---------------------------------------------------------------------

double Linear_model::probability(const vector <double> & c) { 
  
  double f=0;
  
  int ndatapts=data.size();
  for(vector<Point>::const_iterator i=data.begin(); i != data.end();
      i++) { 
    double x=i->x[0];
    double fp=c[0]*x+c[1];
    f-=(i->en-fp)*(i->en-fp)/(2*i->err*i->err);
  }
  return f;
}

//---------------------------------------------------------------------
void Linear_model::niceprint(const vector <double> & c, const vector <double> & cerr) {
  int n=data[0].x.size();
  cout << "Function: c[0]*x+c[1] \n";
  for(int i=0; i< 2; i++) { 
    cout << "c[" <<  i << "] = " << c[i] << " +/- " << cerr[i] << endl;
  }
    
}


//###########################################################################

void Quadratic_model::read(istream & is) { 
  read_into_data(is, data);
}
//------------------------------------------------------------------------

void Quadratic_model::generate_guess(vector <double> & c) { 
  int ndim=data[0].x.size();
  assert(ndim==1);
  int nparms=3;
  c.resize(nparms);
  //cout << " ndim " << ndim << " nparms " << nparms << endl;
  //this is probably not terribly optimal, but it shouldn't be too hard to brute-force our way through it.
  int ndata=data.size();
  c[2]=(data[0].x[0]+data[ndata-1].x[0])*(1+0.1*rng.gasdev())/2.0;
  c[1]=30*(0.5+rng.gasdev()*.1);
  c[0]=data[0].en+0.5*rng.gasdev();
  
}

//---------------------------------------------------------------------

double Quadratic_model::probability(const vector <double> & c) { 
  
  double f=0;
  
  int ndatapts=data.size();
  for(vector<Point>::const_iterator i=data.begin(); i != data.end();
      i++) { 
    double x=i->x[0];
    double fp=c[0]+c[1]*(x-c[2])*(x-c[2]);
    f-=(i->en-fp)*(i->en-fp)/(2*i->err*i->err);
  }
  return f;
}

//---------------------------------------------------------------------

void Quadratic_model::niceprint(const vector <double> & c, const vector <double> & cerr) {
  int n=data[0].x.size();
  cout << "Function: c[0]+c[1]*(x-c[2])**2 \n";
  for(int i=0; i< 3; i++) { 
    cout << "c[" <<  i << "] = " << c[i] << " +/- " << cerr[i] << endl;
  }

  cout << "gnuplot: f(x)=" << c[0] << " + " << c[1] << "*(x-"
    << c[2] << ")**2" << endl;
    
}

//###########################################################################

void Cubic_model::read(istream & is) { 
  read_into_data(is, data);
}
//------------------------------------------------------------------------

void Cubic_model::generate_guess(vector <double> & c) { 
  int ndim=data[0].x.size();
  assert(ndim==1);
  int nparms=4;
  c.resize(nparms);
  //cout << " ndim " << ndim << " nparms " << nparms << endl;
  //this is probably not terribly optimal, but it shouldn't be too hard to brute-force our way through it.
  int ndata=data.size();
  c[3]=0.0;
  c[2]=(data[0].x[0]+data[ndata-1].x[0])*(1+0.1*rng.gasdev())/2.0;
  c[1]=30*(0.5+rng.gasdev()*.1);
  c[0]=data[0].en+0.5*rng.gasdev();
  
}

//---------------------------------------------------------------------

double Cubic_model::probability(const vector <double> & c) { 
  
  double f=0;
  
  int ndatapts=data.size();
  for(vector<Point>::const_iterator i=data.begin(); i != data.end();
      i++) { 
    double x=i->x[0];
    double fp=c[0]+c[1]*(x-c[2])*(x-c[2])
      +c[3]*(x-c[2])*(x-c[2])*(x-c[2]);
    f-=(i->en-fp)*(i->en-fp)/(2*i->err*i->err);
  }
  return f;
}

//---------------------------------------------------------------------

void Cubic_model::niceprint(const vector <double> & c, const vector <double> & cerr) {
  int n=data[0].x.size();
  cout << "Function: c[0]+c[1]*(x-c[2])**2+c[3]*(x-c[2])**3 \n";
  for(int i=0; i< 4; i++) { 
    cout << "c[" <<  i << "] = " << c[i] << " +/- " << cerr[i] << endl;
  }

  cout << "gnuplot: f(x)=" << c[0] << " + " << c[1] << "*(x-"
    << c[2] << ")**2" << " + " << c[3] << "*(x-" << c[2] << ")**3" << endl;
    
}

//###########################################################################

void Centered_quartic::read(istream & is) { 
  read_into_data(is, data);
}
//------------------------------------------------------------------------

void Centered_quartic::generate_guess(vector <double> & c) { 
  int ndim=data[0].x.size();
  assert(ndim==1);
  int nparms=3;
  c.resize(nparms);
  //cout << " ndim " << ndim << " nparms " << nparms << endl;
  //this is probably not terribly optimal, but it shouldn't be too hard to brute-force our way through it.
  int ndata=data.size();
  c[2]=rng.gasdev()*1.0;
  c[1]=rng.gasdev()*1.0;
  c[0]=data[0].en+0.1*rng.gasdev();
  
}

//---------------------------------------------------------------------

double Centered_quartic::probability(const vector <double> & c) { 
  
  double f=0;
  
  int ndatapts=data.size();
  for(vector<Point>::const_iterator i=data.begin(); i != data.end();
      i++) { 
    double x=i->x[0];
    double fp=c[0]+c[1]*x*x+c[2]*x*x*x*x;
    f-=(i->en-fp)*(i->en-fp)/(2*i->err*i->err);
  }
  return f;
}

//---------------------------------------------------------------------

void Centered_quartic::niceprint(const vector <double> & c, 
    const vector <double> & cerr) {
  int n=data[0].x.size();
  cout << "Function: c[0]+c[1]*x^2+c[2]*x^4" << endl;
  for(int i=0; i< 3; i++) { 
    cout << "c[" <<  i << "] = " << c[i] << " +/- " << cerr[i] << endl;
  }

  cout << "gnuplot: f(x)=" << c[0] << " + " << c[1] << "*x**2+"
    << c[2] <<"*x**4" << endl;
    
}

//###################################################################

void Multiple_linear::read(istream & is) { 
  read_into_data(is, data);
}
//------------------------------------------------------------------------

void Multiple_linear::generate_guess(vector <double> & c) { 
  ndim=data[0].x.size()-1;
  nlines=1;
  for(vector<Point>::iterator i=data.begin(); i!=data.end(); i++) { 
    if(nlines < int(i->x[0]+1)) nlines=int(i->x[0]+1);
  }
  int nparms=ndim+nlines;
  c.resize(nparms);
  for(vector<Point>::iterator i=data.begin(); i!=data.end(); i++) { 
    int line=int(i->x[0]);
    c[line]=i->en+rng.gasdev()*10;
  }
  for(int i=nlines; i< nparms; i++) { 
    c[i]=rng.gasdev()*500;
  }
  
}

//---------------------------------------------------------------------

double Multiple_linear::probability(const vector <double> & c) { 
  
  double f=0;
  
  int ndatapts=data.size();
  for(vector<Point>::const_iterator i=data.begin(); i != data.end();
      i++) { 
    int l=i->x[0];
    double fp=c[l];
    
    for(int j=0; j< ndim; j++) { 
      fp+=i->x[j+1]*c[nlines+j];
    }
    f-=(i->en-fp)*(i->en-fp)/(2*i->err*i->err);
  }
  return f;
}

//---------------------------------------------------------------------

void Multiple_linear::niceprint(const vector <double> & c, 
    const vector <double> & cerr) {

  for(int i=0; i< nlines; i++) { 
    cout << "E" << i<< " " << c[i] << " +/- " << cerr[i] << endl;
  }
  for(int i=nlines; i< nlines+ndim; i++) { 
    cout << "C" << i-nlines << " "<< c[i] << "+/-" << cerr[i] << endl;
  }
  cout << "Assessment of the fit:" << endl;

  for(vector<Point>::const_iterator i=data.begin(); i != data.end();
      i++) { 
    int l=i->x[0];
    double fp=c[l];
    
    for(int j=0; j< ndim; j++) { 
      fp+=i->x[j+1]*c[nlines+j];
    }
    cout << "fitted value " << fp << " data " << i->en 
      << " +/- " << i->err << endl;
  }
  

    
}




//---------------------------------------------------------------------
//###########################################################################

void Prob_distribution::avg(double & avg, double & err) { 
  double sum=0;
  for(vector<double>::iterator i=density.begin();
      i!=density.end(); i++) sum+=*i;
  
  avg=0;
  double a=mymin;
  for(vector <double>::iterator i=density.begin();
      i!= density.end(); i++) {
    a+=spacing;
    avg+=a*(*i)/sum;
  }
  
  a=mymin;
  err=0;
  for(vector <double>::iterator i=density.begin();
      i!= density.end(); i++) {
    a+=spacing;
    err+=(a-avg)*(a-avg)*(*i)/sum;
  }
  err=sqrt(err);
}

//---------------------------------------------------------------------

double Prob_distribution::skew() { 
  double mu,sigma;
  avg(mu,sigma);
  
  double sum=0.0;
  for(vector<double>::iterator i=density.begin();
      i!=density.end(); i++) sum+=*i;
  
  double mu3=0.0;
  
  double a=mymin;
  for(vector <double>::iterator i=density.begin(); 
      i!= density.end(); i++) { 
    a+=spacing;
    mu3+=(a-mu)*(a-mu)*(a-mu)*(*i)/sum;
  }
  return mu3/(sigma*sigma*sigma);
}




//###########################################################################


struct Walker { 
  vector <double> c;
  double prob;
  vector <double> grad;
  Walker(int size) { 
    c.resize(size);
    grad.resize(size);
  }
  void update(Prob_model * mod) { 
    prob=mod->probability(c);
    mod->gradient(c,grad);
  }
  
  void updatedir(Prob_model * mod, int d) { 
    prob=mod->probability(c);
    vector <double> tmpc=c;
    double del=1e-12;
    tmpc[d]+=del;
    double px=mod->probability(tmpc);
    grad[d]=(px-prob)/del;
  }
};

//---------------------------------------------------------------------


int check_model_consistency(Prob_model & mod,vector <double> & c) { 
  double del=1e-12;
  int nparms=c.size();
  Walker walk(nparms);
  walk.c=c;
  walk.update(&mod);
  Walker nw=walk;
  for(int p=0; p < nparms; p++) { 
    nw=walk;
    nw.c[p]+=del;
    nw.update(&mod);
    double ratio=nw.prob-walk.prob;
    double deriv=(ratio)/del;
    cout << "numerical derivative " << deriv 
      << " analytic " << walk.grad[p] <<    endl;
    
  }
  
}


//---------------------------------------------------------------------


void sample_min(vector <double> & c, Prob_model & mod) { 
  
  Walker walker(c.size());
  walker.c=c;

  walker.update(&mod);
  Walker best_walker=walker;
  
  int nstep=100000;
  int warmup=nstep/3;
  Walker nw=walker;
  int nparms=c.size();
  vector <double> tmpc;
  vector <double> tstep(nparms);
  for(int p=0; p < nparms; p++) tstep[p]=2e-12;
  double acc=0;
  //cout << "initial probability " << walker.prob << endl;
  vector <double> avgs(nparms);
  vector <double> vars(nparms);
  int navgpts=0;
  for(dit_t i=avgs.begin(); i!= avgs.end(); i++) *i=0.0;
  for(dit_t i=vars.begin(); i!= vars.end(); i++) *i=0.0;
  
  Prob_distribution min1(2000, 0.0, .001);
  Prob_distribution min2(8000, -4.0, .001);

  
  double avg_min=0;
  for(int step=0; step < nstep; step++) { 
    nw=walker;
    for(int p=0; p < nparms; p++) { 
      double ststep=sqrt(tstep[p]);
      double ts=tstep[p];
      double delta=1e-9;
      walker.updatedir(&mod,  p);
      nw.c[p]=walker.c[p]+ststep*rng.gasdev()+ts*walker.grad[p];
      nw.updatedir(&mod, p);
      
      
      double diff=nw.c[p]-walker.c[p];
      double num=-(-diff-ts*nw.grad[p])*(-diff-ts*nw.grad[p]);
      double den=-(diff-ts*walker.grad[p])*(diff-ts*walker.grad[p]);
      double accprob=exp(nw.prob-walker.prob+(num-den)/(2*ts));
      //cout << " acc " << accprob << " num " << num 
      //  << " nw.prob " << nw.prob << " walker.prob  " << walker.prob << endl;
      if(accprob+rng.ulec()> 1.0) {
        acc++;
        walker=nw;
        if(tstep[p] < 1.0) 
          tstep[p]*=1.1;
      }
      else { 
        nw.c=walker.c;
        tstep[p]*=.9;
      }
    }
    
    if(walker.prob > best_walker.prob) best_walker=walker;
    if(step > warmup) { 
      min1.accumulate(walker.c[1],1.0);
      min2.accumulate(walker.c[2],1.0);
      for(int p=0; p < nparms; p++) { 
        double oldavg=avgs[p];
        double oldvar=vars[p];
        avgs[p]=oldavg+(walker.c[p]-oldavg)/(navgpts+1);
        if(navgpts >0) { 
          vars[p]=(1-1.0/navgpts)*oldvar+(navgpts+1)*(avgs[p]-oldavg)*(avgs[p]-oldavg);
        }
      }
      navgpts++;
    }
  }
  cout << "acceptance " << acc/(nstep*nparms) << endl;
  cout << "timesteps   avg   err" << endl;
  for(int p=0; p < nparms; p++) {
    vars[p]=sqrt(vars[p]);
    cout << tstep[p]  << "  " << avgs[p] << "  " << vars[p] << endl;
  }
  cout << endl;
  
  mod.niceprint(avgs, vars);
  
  string nm="minimum";
  min1.write(nm);
  cout << "minimum skew " << min1.skew() << endl;
  string nm2="minimum2";
  min2.write(nm2);
  
  c=best_walker.c;
   
}

//---------------------------------------------------------------------

