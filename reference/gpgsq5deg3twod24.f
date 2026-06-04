      include '/opt/intel/Compiler/11.1/064/mkl/include/mkl_vsl.fi'
      implicit double precision (a-h,o-z)
      double precision X(39),S(39),acc(6),rry(10,6)
      integer mn(6)
      OPEN(1,FILE='/u/roger/twod/geoff/twod24data')
      OPEN(3,STATUS='SCRATCH')

      nsim=16
      ns=4500
      mc=3
      nc=4
      nq=5
      do 1 is=1,nsim
      do 10 i1=1,nc
      read(1,*)acc(1),mn(1),(rry(j,1),j=1,nq),x1,x2,
     *acc(2),mn(2),(rry(j,2),j=1,nq),x1,x2
     *,acc(3),mn(3),(rry(j,3),j=1,nq),x1,x2
C    *,acc(4),mn(4),(rry(j,4),j=1,nq),x1,x2
C     sum=mn(1)+mn(2)+mn(3)
C     acc(1)=mn(1)*1./sum
C     acc(2)=mn(2)*1./sum
C     acc(3)=mn(3)*1./sum
      write(*,"(3(f6.3,i5,5f8.0/))")
     *(acc(i),mn(i),(rry(j,i),j=1,nq),i=1,mc)
      do 31 i=1,mc
   31 write(3,*)acc(i),mn(i),(rry(j,i),j=1,nq)
   10 continue
      rewind (3)
C Ter, st, a, sa
      x(1)=200.
      X(2)=50.
      x(3)=50.
      x(4)=10.
C (s=1), si, sig (gp smooth), range in drift rates (mult by av)
      x(5)=4.
      x(6)=5.
      x(7)=.7
C drift 1, drift 2, drift 1 drift 2 second cond.
      x(8)=20.
      x(9)=10.
      x(10)=60.

      x(11)=50.
      x(12)=0.4
      x(13)=0.8
      x(14)=0.4
      
      NV=10
      sss=5.
      do 9 i3=1,8
      if(i3.gt.4)sss=10.
      CRIT=.0E-7
      ITMAX=40
      ITRACE=10
      IOPT=100
      DO 5 I=1,NV
    5 S(I)=X(I)/sss
C     s(1)=s(1)/25.
C     s(2)=s(2)/25.
      if(i3.lt.3)s(1)=0.
      if(i3.lt.3)s(2)=0.
      if(i3.eq.4)x(2)=x(2)/2.5
C     read(*,*)(X(I),I=1,NV),dum
      WRITE(*,47)(X(I),I=1,NV)
      CALL SIMPLX (X,S,CRIT,ITMAX,ITRACE,IOPT,NV)
C     yy=fofs(nv,x)
    9 continue
      CRIT=1.0E-7
      itmax=200
      iopt=400
      DO 6 I=1,NV
    6 S(I)=X(I)/10.0
      CALL SIMPLX (X,S,CRIT,ITMAX,ITRACE,IOPT,NV)

      WRITE(*,47)(X(I),I=1,NV)
   47 FORMAT(32F8.3)
    1 continue
      STOP
      END

      DOUBLE PRECISION FUNCTION FOFS(NV,X)
      implicit double precision (a-h,o-z)
      double precision X(NV),pxy(6),acc(6),co(6),rtm(6),cc(5)
      double precision qc(10,6),rry(10,6),pq(10),pqq(10),c1(6),pp(6)
      integer mn(6)

      ncut=8
      mc=3
      nc=4
      nq=5

      do 3 j=1,nq
      pqq(j+1)=.2
    3 pq(j)=.5/nq+(j-1.)/nq
      pqq(1)=.1
      pqq(nq+1)=.1
      mx=0
C     if(x(10).gt.50.)x(10)=50.
C     if(x(10).lt.5.)x(10)=5.
C     si=x(10)
C     if(x(11).gt.50.)x(11)=50.
C     sv=x(11)
      if(x(1).lt.175.)x(1)=175.
C     if(x(11).lt.5.)x(11)=5.
      ter=x(1)
      if(x(2).lt.20.)x(2)=20.
      if(x(2).gt.ter*1.5)x(2)=ter*1.5
      st=x(2)
C     if(x(3).gt.8.)x(3)=8.
      if(x(3).lt.1.)x(3)=1.
      a=x(3)
      if(x(4).gt.x(3)/2.0)x(4)=x(3)/2.0
      if(x(4).lt.0.01)x(4)=0.01
      sa=x(4)
C     if(x(5).lt.5.0)x(5)=5.0
      si=x(5)
      if(x(6).lt.1.0)x(6)=1.0
C     if(x(6).gt.x(9)/2.)x(6)=x(9)/2.
      sig=x(6)
      if(x(7).lt..3)x(7)=.3
      sv=x(7)
C     print*,"par",(x(i),i=1,8)

      mx=0
      chi=0.
      do 2 ij=1,nc

      do 11 i1=1,mc
      read(3,*)acc(i1),mn(i1),(rry(j,i1),j=1,nq)
   11 continue

      i3=7+ij
      if(ij.gt.2)i3=5+ij
      if(x(i3).lt.0.01)x(i3)=0.01
      av=x(i3)
      if(ij.gt.2)a=x(10)
C     print*,ij,i3,av,a

      call accum(ter,st,a,sa,si,sig,av,sv,pxy,qc,rtm,rry)
      mmn=0
      do 10 i1=1,mc
      mmn=mmn+mn(i1)
   10 continue

C     write(*,"(3(f6.3,i5,5f8.0))")
C    *(acc(i1),mn(i1),(rry(j,i1),j=1,nq),i1=1,mc)
C     write(*,"(3(f6.3,i5,5f8.3))")
C    *(pxy(i1),mn(i1),(qc(j,i1),j=1,nq),i1=1,mc)

      do 2 i=1,mc
      if(mn(i).ge.ncut.and.mx.eq.0)then
      chi=chi+mmn*acc(i)*pqq(1)*log(acc(i)*pqq(1)/
     *(pxy(i)*qc(1,i)+.00001))
      do 45 j=2,nq
      yy=qc(j,i)-qc(j-1,i)
      if(yy.lt.0.001)yy=0.001
      chi=chi+mmn*acc(i)*pqq(j)*log(acc(i)*pqq(j)/(pxy(i)*yy+.00001))
   45 continue
      chi=chi+mmn*acc(i)*pqq(nq+1)*log(acc(i)*pqq(nq+1)/
     *(pxy(i)*(1.-qc(nq,i))+.00001))
      elseif(mn(i).lt.ncut)then 
      chi=chi+mmn*(acc(i)+.002)*log((acc(i)+.002)/pxy(i))
      endif

C     chi=chi+mmn*(pxy(i)*qc(1,i)-acc(i)*pqq(1))*
C    *(pxy(i)*qc(1,i)-acc(i)*pqq(1))/(pxy(i)*qc(1,i)+.00001)
C     do 5 j=2,nq
C     if(yy.lt.0.001)yy=0.001
C     chi=chi+mmn*(pxy(i)*yy-acc(i)*pqq(j))*(pxy(i)*yy-acc(i)*pqq(j))/
C    *(pxy(i)*yy+.00001)
C     print*,chi,pxy(i),qc(j,i),acc(i)
C   5 continue
C     chi=chi+mmn*
C    *(pxy(i)*(1.-qc(nq,i))-acc(i)*pqq(nq+1))*(pxy(i)*(1.-qc(nq,i))-
C    *acc(i)*pqq(nq+1))/(pxy(i)*(1.-qc(nq,i))+.00001)
C     print*,chi
C     elseif(mn(i).gt.0.and.mn(i).lt.5.and.mx.eq.1)then
C     yy=qc(3,i)
C     chi=chi+(pxy(i)*yy-acc(i)*0.5)*(pxy(i)*yy-acc(i)*0.5)/
C    *(pxy(i)*yy+.00001)
C     chi=chi+(pxy(i)*(1.-yy)-acc(i)*0.5)*(pxy(i)*(1.-yy)-acc(i)*0.5)
C    */(pxa*(1.-yy)+.00001)
C     elseif(mn(i).lt.6)then
C     chi=chi+mmn*(pxy(i)-acc(i))**2/pxy(i)
C     endif
    2 continue
      rewind (3)
      fofs=chi
      RETURN
      end

      subroutine accum(ter,st,cr,crsd,si,sig,av,as,pxy,qc,rtm,rry)
C si=sd in stim (v), sig=gp smooth
      implicit double precision (a-h,o-z)
      integer n,nt1,nt2,nt3,nt4,iseed,d(1)
      parameter (n=72,nsim=4000,nstep=400)
      integer ich(nsim),irt(nsim),lc(10000),lk(6),l2
      double precision x(n),r(n,n),v(n),y(n),g(100000),ze,one,scal,
     *a(n),b(n),gs(nsim),gg(505),gu1(nsim),grr(nstep*n+5),gr(n),
     *rt(10000),jk(6),rc(10000,6),re(10000),rry(10,6),pxy(6),
     *rrn(10),cor(6),rtm(6),rr(10,6),qc(10,6),qe(10),pk(6),qeq(10)
      integer brng
      method=0
      call gettim(nt1,nt2,nt3,nt4)
      iseed=(nt4*100000+nt3*10000+nt4*100+nt3)+1
      call ranunif(gs,nsim,iseed)
      pi=4.0*atan(1.0)
      scal=1.0D0
C mean (if needed SD of drift rate distribution), 10 ms steps
      mc=3
      nq=5
      u=180./5
      e=10.
      ipa=150/5
      ipb=210/5
      ipc=100/5
      ipd=260/5
      ze=0.0D0
      one=1.D0
      do 1 i=1,n
      x(i)=i
      y(i)=0.0
      a(i)=0.0
      b(i)=0.0
C v is drift rate distrib, a is accumulated evidence array, b just to
C save some evidence across the 200 simulated runs.
    1 v(i)=av*exp(-(float(i)-u)**2/(2.*si**2))/(si*sqrt(2.*pi))
      call gp(n,x,sig,r)
      do 6 kk=1,nsim
      irt(kk)=0
    6 ich(kk)=0

C m is number of simulations irt is RT on the trial and ich posn (1-360)
!$omp parallel
!$omp do private(gg,i,npp,npos,k,icon,j,terr,iseed,ii,
     * gu1,a,y,am,d,istop,ac,asum,it,b,gr,grr,ij,icon2,crr)
      do ii=1,nsim
       iseed=gs(ii)*99999999
       if(iseed.lt.0)iseed=-iseed
       call ranunif(gu1,n,iseed)
      do 8 i=1,n
      y(i)=0.0
    8 a(i)=0.0
      crr=cr+crsd*(gu1(5)-.5)
       npp=nstep*n+5
C     print*,iseed
       call gauss(ze,one,grr,npp,iseed)
      do 2 j=1,nstep
      do 31 i=1,n
   31 gr(i)=grr(i+(j-1)*n+5)
C     write(*,"(11f6.1)")gr
C     print*,i+(j-1)*n+5
      CALL DGEMM('T','N',n,1,n,scal,r,n,gr,n,0.D0,y,n)
      asum=0.0D0
      ssv=sv*(gu1(8)-.5)
C     if(j.lt.5.and.k.eq.1)write(*,"(9f7.2)")(y(ii),ii=1,n)
      do 3 i=1,n
C accumulation of evidence .8 drift plu 2 time noise (I can scale the .8
C and the criterion so there is only one param)
      a(i)=a(i)+(1.+ssv)*v(i)+5.*y(i)
      asum=asum+a(i)
    3 continue
      asum=asum/(n*1.0)
      do 9 i=1,n
    9 a(i)=a(i)-asum
      am=0
      istop=0
      am=maxval(a)
      d=maxloc(a)
      istop=d(1)
      if(am.gt.crr)go to 4
C finish decision process if the largest value is greater than cr
    2 continue
    4 continue
C increment array of responses over position. Maybe add a time. And then
C need a N array to find average times.
      do 10 i=1,n
   10 b(i)=b(i)+a(i)/(nsim*1.)
C Save values of position and number of iterations (RT)
      irt(ii)=j
      ich(ii)=istop
      rt(ii)=j+(ter+st*(.5-gu1(10)))/e
      lc(ii)=2
C this order is correct from new file from Russ twod13data3603
      if(istop.gt.ipa.and.istop.lt.ipb)lc(ii)=1
      if(istop.le.ipc.or.istop.ge.ipd)lc(ii)=3
C     write(4,"(4i4,9f7.2)")ii,j,ich(ii),lc(ii),rt(ii)
      enddo
!$omp end parallel
       do 28 i=1,10000
       do 28 j=1,6
       rc(i,j)=0.
   28 continue
       do 29 i=1,mc
       jk(i)=0
   29  lk(i)=0

       do 11 ii=1,nsim
       jjj=lc(ii)
       lk(jjj)=lk(jjj)+1
       l2=rt(ii)
       jk(jjj)=jk(jjj)+l2
       rc(l2,jjj)=rc(l2,jjj)+1.
C      print*,ii,l2,jjj,lk(jjj),jk(jjj),rc(l2,jjj)
   11 continue
      lsum=0
      do 13 j=1,mc
   13 lsum=lsum+lk(j)
      do 14 i=1,mc
       cor(i)=lk(i)*1.0/(lsum*1.0)
   14  rtm(i)=e*jk(i)/(1.*lk(i))
C     write(4,"(9f7.2)")rtm,cor
       do 15 j=1,mc
       do 5 i=1,nstep
       if(lk(j).ne.0)rc(i,j)=rc(i,j)/(lk(j)*1.0)
      re(i)=rc(i,j)
    5 continue
C     write(*,"(9f7.2)")(re(i),i=1,200)
      do 7 i=1,nq
      rrn(i)=rry(i,j)
      qe(i)=(i-1.0)/nq+.5/nq
      qeq(i)=qe(i)
    7 continue
      tr=0.
      call qant(re,nstep,qeq,nq,e,tr)
      corr=cor(j)
      pxy(j)=cor(j)
C     write(*,"(9f7.2)")corr,(qeq(i),i=1,nq)
      tr=0.
C     write(*,"(9f7.2)")(rry(i,1),i=1,5)
C     write(*,"(9f7.2)")(rrn(i),i=1,5)
      call qant1(re,nstep,rrn,qe,nq,e,tr,corr)
C     write(*,"('qants',9f7.2)")qe
      do 16 i=1,nq
   16 qc(i,j)=qe(i)
   15 continue
      return
      end

      subroutine qant1(r,n,rt,q,m,e,ter,corr)
      implicit double precision (a-h,o-z)
      double precision r(n),q(m),rr(n),rt(m)
      do 3 i=1,n
    3 rr(i)=0.
      do 2 i=2,n
    2 rr(i)=rr(i-1)+r(i)
      rsum=rr(n)
      do 5 i=1,n
    5 rr(i)=rr(i)/rsum
C now rr contains the cumulative distribution function (0 to 1)
C now need to get probability given RT.  RT=e*step+ter
      do 4 j=1,m
      art=(rt(j)-ter)/e
      irt=art
      if(irt.lt.1)irt=1
      if(irt.lt.1)art=1.
      if(irt.ge.n)irt=n-1
      q(j)=rr(irt)+(art-1.*irt)*(rr(irt+1)-rr(irt))
      if(q(j).lt.0.01)q(j)=0.01
    4 continue
      return
      end

      subroutine qant(r,n,q,m,e,ter)
      implicit double precision (a-h,o-z)
      double precision r(n),q(m),rr(n)
      do 3 i=1,n
    3 rr(i)=0.
      rr(1)=r(1)
      do 2 i=2,n
    2 rr(i)=rr(i-1)+r(i)
C now rr contains the cumulative distribution
      j=1
      do 1 i=1,n-1
C     print*,i,j,rr(i),q(j)
      if(rr(i).gt.q(j))then
      rrr=rr(i)-rr(i-1)
      if(rrr.lt.0.000001)rrr=0.00001
      q(j)=e*(i-1+(q(j)-rr(i-1))/(rrr))+ter
      j=j+1
      if(j.gt.m)go to 4
      endif
    1 continue
    4 continue
      return
      end

      subroutine gauss(uu,ss,ggg,ng,seed)
      USE MKL_VSL_TYPE
      USE MKL_VSL
      double precision uu,ss,ggg(6100),u,s
      integer brng,method,seed,ng
      TYPE (VSL_STREAM_STATE) :: stream
      brng=VSL_BRNG_MCG31
      method=0
      m=1
      u=uu
      s=ss
      ierr=vslnewstream( stream, brng,  seed )
        ierr=vdrnggaussian( method, stream, ng, ggg, u, s)
      errcode=vsldeletestream( stream )
      return
      end

      subroutine ranunif(gu,nit,seed)
      USE MKL_VSL_TYPE
      USE MKL_VSL
      double precision gu(21000),q1,q2
      integer brng,method,seed
      TYPE (VSL_STREAM_STATE) :: stream
      brng=VSL_BRNG_MCG31
      method=0
      m=1
      q1=0.D0
      q2=1.D0
      ierr=vslnewstream( stream, brng,  seed )
        ierr=vdrnguniform( method, stream, nit, gu, q1,q2)
C     print*,idum,kg(1),kg(2),kg(3),kg(4),"rand"
      errcode=vsldeletestream( stream )
      return
      end

      subroutine gp(n,x,sig,r)
      implicit double precision (a-h,o-z)
      double precision x(n),k(n,n),r(n,n)
      do 2 i=1,n
      do 2 j=1,n
      r(i,j)=0.0D0
      k(i,j)=x(j)-x(i)
      k(i,j)=1.0D0*exp(-0.5D0*k(i,j)*k(i,j)/(sig*sig))
      if(i.eq.j)k(i,j)=k(i,j)+1.0D-12
    2 continue
      call DPOTF2('Upper',n,k,n,info)
      if(info.gt.0)print*,"problem, not positive definite"
C Get upper right triangular matrix.
      do 1 i=1,n
        do 1 j=i,n
   1      r(i,j)=k(i,j)
      return
      end

      SUBROUTINE SIMPLX(X,SCALE,CRIT,ITMAX,ITRACE,IOPT,NV)
      implicit double precision (a-h,o-z)
C     X IS PARAMETER VALUES, SCALE DETERMINES SPREAD OF SIMPLX POINTS
C     AROUND STARTING VALUE, NV IS NO. OF PARAMETERS
C     CRIT IS CONVERGENCE CRITERION, ITMAX IS MAX. NO. OF ITERATIONS,
C     AND ITRACE GIVES PRINT OUT EVERY ITRACE ITERATIONS,
C     EVERY IOPT TRIALS TEST IF BEST PARAMETERS WITHIN .00001
C     PROGRAM IS DIMENSIONED FOR A MAXIMUM OF 9 PARAMETERS
      double precision X(39),SCALE(39),PL(40)
      DIMENSION P(40,39),Y(40),PSTAR(39),PBAR(39)
      INTEGER MATRIX(5,5)
C     THIS MATRIX COUNTS THE NUMBER OF EACH TYPE OF MOVE
C     FOLLOWING EACH TYPE OF PREVIOUS MOVE SUMMING MATRIX(I,J)
C     OVER I=1,5 GIVES THE NUMBER OF TIMES MOVE J WAS USED
      LOGICAL TRACE
      ALPH=1.0
      BET=0.5
      TRACE=.TRUE.
      GAMMA=2.0
      IF(CRIT.LE.0.0)CRIT=1.E-8
      IF(ITMAX.LE.0)ITMAX=1
      DO 5 I=1,5
      DO 5 J=1,5
    5 MATRIX(I,J)=0
      LASTM=1
      ITER=1
      NVP1=NV+1
      FNV=NV
      FNVP1=NVP1
      ASSIGN 35 TO NEXT
C     DO 15 J=1,NV
C     IF(SCALE(J).GT.0.0) GO TO 15
C     SCALE(J)=X(J)*.2
C  15 CONTINUE
C     GENERATE A REGULAR SIMPLX IN NV DIMENSIONS WITH VERTICES AT UNIT
C     DISTANCE FROM THE CENTROID AND CENTROID AT THE ORIGIN
      T1=(1.0-SQRT(FNVP1))/SQRT(FNV**3)
      T2=SQRT(FNVP1/FNV)+T1
      DO 23 I=1,NV
      DO 22 J=1,NV
      IF (I-J)20,21,20
   20 P(I,J)=T1
      P(J,I)=T1
      GO TO 22
   21 P(I,I)=T2
   22 CONTINUE
   23 CONTINUE
      T1=-1.0/SQRT(FNV)
      DO 24 J=1,NV
   24 P(NVP1,J)=T1
C     NOTE THAT THE NV POINTS P(I,J), J=1,NV DEFINE THE ITH
C     POINT IN THE NV DIMENSIONED SIMPLEX
C     MOVE CENTROID TO STARTING VECTOR AND SCALE COLUMNS
      DO 26 J=1,NV
      DO 25 I=1,NVP1
   25 P(I,J)=P(I,J)*SCALE(J)+X(J)
   26 CONTINUE
C     COMPUTE FUNCTION VALUES FOR SIMPLEX POINTS AND FIND MIN VALUE
C     OF FUNCTION
      DO 29 I=1,NVP1
      WRITE(*,669) I,(P(I,J),J=1,NV)
      DO 28 J=1,NV
   28 PSTAR(J)=P(I,J)
      T1=FOFS(NV,PSTAR)
   29 Y(I)=T1
   30 T1=Y(1)
      IL=1
      DO 32 I=2,NVP1
      IF(T1-Y(I))32,32,31
   31 T1=Y(I)
      IL=I
   32 CONTINUE
      GO TO NEXT,(35,80)
   35 IF(TRACE) WRITE(*,600)
  600 FORMAT(' ',24X,'NO',4X,'IL',3X,'FUNCTION',4X,'CRITERION',3X,'PARAM
     *ETERS')
      ASSIGN 80 TO NEXT
C     STARTING POINT OF ITERATIVE CYCLE, FIND Y(IH)
 1000 T1=Y(1)
      IH=1
      TRACE=.FALSE.
      IT=ITER-1
      IF(MOD(IT,ITRACE).EQ.0) TRACE=.TRUE.
      IF (MOD(IT,IOPT).NE.0) GO TO 44
      IF (IT.EQ.0) GO TO 43
      DO 41 I=1,NV
      IF (ABS(P(IL,I)-PL(I)).GT.0.00000001) GO TO 43
   41 CONTINUE
      WRITE(*,700) IOPT
  700 FORMAT(/,'NO IMPROVEMENT IN',I3,'TRIALS')
      GO TO 90
   43 DO 42 I=1,NV
   42 PL(I)=P(IL,I)
   44 DO 46 I=2,NVP1
      IF(T1-Y(I))45,46,46
   45 T1=Y(I)
      IH=I
   46 CONTINUE
C     COMPUTE CENTROID EXCLUDING POINT WITH MAX VALUE
      DO 50 J=1,NV
      T1=0.0
      DO 49 I=1,NVP1
      IF(I-IH)48,49,48
   48 T1=P(I,J)+T1
   49 CONTINUE
   50 PBAR(J)=T1/FNV
C     TRY A REFLECTION
C     TRY A REFLECTION
      DO 51 J=1,NV
   51 PSTAR(J)=(1.0+ALPH)*PBAR(J)-ALPH*P(IH,J)
      T1=FOFS(NV,PSTAR)
      IF(T1.GT.Y(IL)) GO TO 54
C     REFLECTION SUCCEEDED, TRY AN EXPANSION
      DO 52 J=1,NV
      P(IH,J)=PSTAR(J)
   52 PSTAR(J)=(1.0+GAMMA)*PSTAR(J)-GAMMA*PBAR(J)
      T2=T1
      T1=FOFS(NV,PSTAR)
      IL=IH
      IF(T1.GT.T2) GO TO 76
C     EXPANSION SUCCEEDED
      IF(TRACE) WRITE(*,601)
  601 FORMAT(20H EXPANSION SUCCEEDED)
      MATRIX(LASTM,1)=MATRIX(LASTM,1)+1
      LASTM=1
      GO TO 72
   54 T2=Y(IL)
      I2=IL
C     REFLECTION FAILED, FIND NEXT TO BIGGEST Y AND TEST
      DO 57 I=1,NVP1
      IF(I-IH)55,57,55
   55 IF(Y(I)-T2)57,57,56
   56 T2=Y(I)
      I2=I
   57 CONTINUE
      IF(T1.LT.T2) GO TO 71
C     NEW POINT CLOSE TO MAXIMUM, EXCHANGE IF NECESSARY AND TRY A CONTRA
C     CTION
      IF(T1.GE.Y(IH)) GO TO 61
C     EXCHANGE PSTAR AND Y
      DO 60 J=1,NV
      T2=PSTAR(J)
      PSTAR(J)=P(IH,J)
   60 P(IH,J)=T2
C     CALCULATE NEW POINT FOR CONTRACTION
   61 DO 62 J=1,NV
   62 PSTAR(J)=(1.0-BET)*PBAR(J)+BET*P(IH,J)
      T1=FOFS(NV,PSTAR)
      IF(T1.LT.Y(IH)) GO TO 70
C     CONTRACTION FAILED
      IF(TRACE) WRITE(*,602)
  602 FORMAT(19H CONTRACTION FAILED)
      MATRIX(LASTM,5)=MATRIX(LASTM,5)+1
      LASTM=5
      DO 66 I=1,NVP1
      IF(I-IL)64,66,64
   64 DO 65 J=1,NV
      P(I,J)=(P(IL,J)+P(I,J))/2.0
   65 PSTAR(J)=P(I,J)
      Y(I)=FOFS(NV,PSTAR)
   66 CONTINUE
      GO TO 30
   70 IF(TRACE) WRITE(*,603)
  603 FORMAT(22H CONTRACTION SUCCEEDED)
      MATRIX(LASTM,4)=MATRIX(LASTM,4)+1
      LASTM=4
      IF(T1.LT.Y(IL))IL=IH
      GO TO 72
   71 IF(TRACE) WRITE(*,664)
  664 FORMAT(12H NORMAL MOVE)
      MATRIX(LASTM,3)=MATRIX(LASTM,3)+1
      LASTM=3
   72 Y(IH)=T1
   73 DO 74 J=1,NV
   74 P(IH,J)=PSTAR(J)
      GO TO 80
   76 IF(TRACE) WRITE(*,604)
  604 FORMAT(21H REFLECTION SUCCEEDED)
      MATRIX(LASTM,2)=MATRIX(LASTM,2)+1
      LASTM=2
      Y(IH)=T2
   80 T1=0.0
C     CHECK FOR MINIMUM
      DO 81 I=1,NVP1
   81 T1=Y(I)+T1
      T1=T1/FNVP1
      T2=0.0
      DO 82 I=1,NVP1
   82 T2=(Y(I)-T1)**2+T2
      T2=SQRT(T2/FNV)
      IF(T2.LT.CRIT) GO TO 90
      IF(TRACE) WRITE(*,665)ITER,IL,Y(IL),T2,(P(IL,J),J=1,NV)
  665 FORMAT(1H+,20X,2I6,2E12.4,39F8.3)
      ITER=ITER+1
      IF(ITER.LT.ITMAX) GO TO 1000
      WRITE(*,666)
  666 FORMAT(29H MAXIMUM NUMBER OF ITERATIONS)
   90 DO 91 J=1,NV
   91 X(J)=P(IL,J)
      WRITE(*,876)((MATRIX(I,J),J=1,5),I=1,5)
  876 FORMAT(/,29X,'SUBSEQUENT MOVE'/' ',24X,'EXP REF NOR CON FLD'
     */' ',17X,'EXP',5I5/' ',17X,'REF',5I5/' ','PREVIOUS',9X,'NOR',5I5/'
     * ','MOVE',11X,'CON',5I5/' ',17X,'FLD',5I5)
      DO 92 I=1,NVP1
      WRITE(*,669) I,(P(I,J),J=1,NV)
  669 FORMAT(' ','VERTEX',I3,2X,39F8.3)
   92 CONTINUE
      WRITE(*,667) ITER,IL,Y(IL),T2
  667 FORMAT(/,'ITER=',I4,1X,'IL=',I2,1X,'Y(IL)=',E9.4,3X,'CRITE
     *',E9.4)
      WRITE(*,668) (P(IL,J),J=1,NV),Y(IL)
      if(iter.ne.200)
     *print*,"a ter eta sz 4drifts (then 2 dummy drifts) z po st"
      if(iter.ne.500)WRITE(*,"(20f10.4)") (P(IL,J),J=1,NV),Y(IL)
      if(iter.ne.40)WRITE(53,"(20f10.4)") (P(IL,J),J=1,NV),Y(IL)
  668 FORMAT(' ','BEST PARAMETER ESTIMATES',39F10.3)
      RETURN
      END


