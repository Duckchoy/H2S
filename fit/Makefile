
OBJS:= Prob_model.o ulec.o macopt.o
HEADERS=Prob_model.h ulec.h macopt.h Point.h Min.h
CXXFLAGS:=-O2

all: fit

fit: $(OBJS) $(HEADERS) fit.o
	$(CXX) -o $@ $(OBJS) fit.o

clean :
	rm -f $(OBJS) fit.o
