//--------------------------------------------------------------------------
// include/ulec.h
//
//
#ifndef ULEC_H_INCLUDED
#define ULEC_H_INCLUDED
#include <cmath>


/*!
 
*/
class Random_generator
{
public:
  Random_generator()
  {
    is1=12345;
    is2=56789;
    iset=0;
    gset=.1;
  }

  void seed(long int is1_,long int is2_)
  {
    is1=is1_;
    is2=is2_;
  }


  void getseed(long int & is1_, long int & is2_) {
    is1_=is1;
    is2_=is2;
  }

  /*!
       uniform random number generator (combined type)

        P. L'Ecuyer, Commun. ACM, 31(1988)742

        portable (32 bits arithmetics)   
   */
  double ulec()
  {
    long int k,iz;
    k=is1/53668;
    is1=is1-k*53668;
    is1=40014*is1-k*12211;
    if (is1<0)
      is1 +=2147483563;
    k=is2/52774;
    is2=is2-k*52774;
    is2=40692*is2-k*3791;
    if (is2<0)
      is2 +=2147483399;
    iz=is1-is2;
    if (iz<0)
      iz +=2147483562;


    //cout << "random number " << 4.656613057e-10*iz << endl;
    return (4.656613057e-10*iz);
  }

  /*!
    Gaussian distributed random number
  */
  double gasdev()
  {

    if (iset ==0)
    {
      double v1, v2, r, fac;
      do
      {
        v1=2.*ulec() -1.;
        v2=2.*ulec() -1.;
        r=v1*v1 + v2*v2;
      }
      while ( r >= 1. || r == 0.);
      fac=sqrt(-2.*log(r)/r);
      gset=v1*fac;
      iset=1;
      //if(mpi_info.node==0) cout << "gasdev  " << v2*fac << endl;
      return v2*fac;
    }
    else
    {
      iset=0;
      //if(mpi_info.node==0) cout << "gasdev " << gset << endl;
      return gset;
    }
  }

private:
  long int is1;
  long int is2;
  int iset;
  double gset;

};

/*!
This is one of the few times that we actually
_want_ a global variable...
*/
extern Random_generator rng;

double unif();

double ranr2exponential();
double rancos();

#endif // ULEC_H_INCLUDED
//--------------------------------------------------------------------------
