#include <iostream>
#include <pthread.h>
#include <windows.h>
#include<semaphore.h>
#include<stdio.h>
#include<time.h>
#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h> //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h> //SSSE4.2

using namespace std;
int n=50;
int thread_count=7;
int thread_num=thread_count+1;
float **A;
float **B;
float **C;
float **D;
float **E;
typedef struct
{
    int t_id;//线程id
    int k;//消去的轮次
}thread_art;
void init(int n)
{
    A=new float*[n];
    for(int i=0;i<n;i++)
        A[i]=new float[n];
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<i;j++)
            A[i][j]=0.0;
        A[i][i]=1.0;
        for(int j=i+1;j<n;j++)
            A[i][j]=rand();
    }
    for(int i=1;i<n;i++)
    {
        for(int j=0;j<n;j++)
            A[i][j]+=A[i-1][j];
    }


}
void deepcopy()
{
    B=new float*[n];
    for(int i=0;i<n;i++)
        B[i]=new float[n];
    C=new float*[n];
    for(int i=0;i<n;i++)
        C[i]=new float[n];
    D=new float*[n];
    for(int i=0;i<n;i++)
        D[i]=new float[n];
    E=new float*[n];
    for(int i=0;i<n;i++)
        E[i]=new float[n];
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<n;j++)
        {
            B[i][j]=A[i][j];
            C[i][j]=A[i][j];
            D[i][j]=A[i][j];
            E[i][j]=A[i][j];
        }
    }
}
void chuanxing()
{
    long long head,tail,freq ; // timers
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq );
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    for (int k = 0; k < n; k++)
    {
		for (int j = k + 1; j < n; j++)
		{
			A[k][j] /= A[k][k];
		}
		A[k][k] = 1.0;
		for (int i = k + 1; i < n; i++)
		{
			for (int j = k + 1; j < n; j++)
			{
				A[i][j] -= A[i][k] * A[k][j];
			}
			A[i][k] = 0;
		}

	}

	QueryPerformanceCounter((LARGE_INTEGER *)&tail );
	cout <<(tail-head)*1000.0/freq<<"          ";

}
void *thread_funcd(void *parm)
{
    thread_art*p=(thread_art*)parm;
    int k=p->k;
    int id=p->t_id;
    for(int i=k+id+1;i<n;i+=thread_count)
    {
        __m128 vaik=_mm_set1_ps(C[i][k]);
            int j=k+1;
            for(j=k+1;j+4<=n;j+=4)
            {
                __m128 vakj=_mm_loadu_ps(&C[k][j]);
                __m128 vaij=_mm_loadu_ps(&C[i][j]);
                __m128 vx=_mm_mul_ps(vakj,vaik);
                vaij=_mm_sub_ps(vaij,vx);

                _mm_storeu_ps(&C[i][j], vaij);

            }
            while(j<n)
            {
                C[i][j]-=C[k][j]*C[i][k];
                j++;
            }
            C[i][k]=0;
    }
    pthread_exit(nullptr);
}
void dynamic_simd()
{
    long long head,tail,freq ; // timers
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq );
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    thread_count=7;
    int k;
    for (k = 0; k<n; k++)
    {
		for (int j = k + 1; j < n; j++)
		{
			C[k][j] /= C[k][k];
		}
		C[k][k] = 1.0;
        thread_art*param=new thread_art[thread_count];
        pthread_t*handles=new pthread_t[thread_count];
        for(int i=0;i<thread_count;i++)
        {
            param[i].t_id=i;
            param[i].k=k;

        }
        for(int i=0;i<thread_count;i++)
        {
            pthread_create(&handles[i],nullptr,thread_funcd,(void*)&param[i]);
        }
        for(int i=0;i<thread_count;i++)
        {
            pthread_join(handles[i],nullptr);
        }

	}

	QueryPerformanceCounter((LARGE_INTEGER *)&tail );
	cout <<(tail-head)*1000.0/freq<<"          ";
}
sem_t sem_main;
sem_t *sem_workerstart=new sem_t[thread_count];
sem_t *sem_workerend=new sem_t[thread_count];
//sem_t sem_workerstart;
//sem_t sem_workerend;
void *thread_func(void *parm)
{
    thread_art*p=(thread_art*)parm;
    int id=p->t_id;
    for(int k=0;k<n;++k)
    {
        sem_wait(&sem_workerstart[id]);
        for(int i=k+1+id;i<n;i+=thread_count)
        {
             __m128 vaik=_mm_set1_ps(B[i][k]);
            int j=k+1;
            for(j=k+1;j+4<=n;j+=4)
            {
                __m128 vakj=_mm_loadu_ps(&B[k][j]);
                __m128 vaij=_mm_loadu_ps(&B[i][j]);
                __m128 vx=_mm_mul_ps(vakj,vaik);
                vaij=_mm_sub_ps(vaij,vx);

                _mm_storeu_ps(&B[i][j], vaij);

            }
            while(j<n)
            {
                B[i][j]-=B[k][j]*B[i][k];
                j++;
            }
            B[i][k]=0;
        }
        sem_post(&sem_main);
        sem_wait(&sem_workerend[id]);
    }
    pthread_exit(nullptr);
}
void signal()
{
    long long head,tail,freq ; // timers
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq );
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    int flag=sem_init(&sem_main,0,0);
    if(flag!=0)
    {
        cout<<"error";
        return;
    }
    for(int i=0;i<thread_count;i++)
    {
        int flag1=sem_init(&sem_workerstart[i],0,0);
        int flag2=sem_init(&sem_workerend[i],0,0);
        if(flag1==-1||flag2==-1)
        {
            cout<<"error";
            return;
        }
    }
    //sem_init(&sem_workerend,0,0);
    pthread_t *handles=new pthread_t[thread_count];
    thread_art *param=new thread_art[thread_count];
    for(int i=0;i<thread_count;i++)
    {
        param[i].t_id=i;
        pthread_create(&handles[i],nullptr,thread_func,(void*)&param[i]);
    }
    for(int k=0;k<n;++k)
    {
        for(int j=k+1;j<n;j++)
            B[k][j]/=B[k][k];
        B[k][k]=1.0;
        for(int t_id=0;t_id<thread_count;t_id++)
        {
            //cout<<"开始第"<<k<<"轮第"<<t_id<<"消去"<<endl;
            sem_post(&sem_workerstart[t_id]);
            //sem_wait(&sem_main);
        }
        for(int i=0;i<thread_count;i++)
            sem_wait(&sem_main);
        for(int i=0;i<thread_count;i++)
        {
            //cout<<"开始下一轮"<<endl;
            sem_post(&sem_workerend[i]);
        }
    }
    for(int i=0;i<thread_count;i++)
        pthread_join(handles[i],nullptr);
    //sem_destroy(&sem_workerend);
    for(int i=0;i<thread_count;i++)
        sem_destroy(&sem_workerstart[i]);
    sem_destroy(&sem_main);
    QueryPerformanceCounter((LARGE_INTEGER *)&tail );
//	cout <<"静态"<<(tail-head)*1000.0/freq<<"ms"<<endl;
cout <<(tail-head)*1000.0/freq<<"            ";
}
//静态三循环D
sem_t sem_leader;
sem_t *sem_Division=new sem_t[thread_num-1];
sem_t *sem_Elimination=new sem_t[thread_num-1];
void *thread_func2(void *parm)
{
    thread_art*p=(thread_art*)parm;
    int id=p->t_id;
    for(int k=0;k<n;++k)
    {
        if(id==0)
        {
            for(int j=k+1;j<n;j++)
                D[k][j]/=D[k][k];
            D[k][k]=1.0;
        }
        else
            sem_wait(&sem_Division[id-1]);
        if(id==0)
        {
            for(int i=0;i<thread_num-1;i++)
                sem_post(&sem_Division[i]);
        }

            for(int i=k+1+id;i<n;i+=thread_num)
            {
                 __m128 vaik=_mm_set1_ps(D[i][k]);
            int j=k+1;
            for(j=k+1;j+4<=n;j+=4)
            {
                __m128 vakj=_mm_loadu_ps(&D[k][j]);
                __m128 vaij=_mm_loadu_ps(&D[i][j]);
                __m128 vx=_mm_mul_ps(vakj,vaik);
                vaij=_mm_sub_ps(vaij,vx);

                _mm_storeu_ps(&B[i][j], vaij);

            }
            while(j<n)
            {
                D[i][j]-=D[k][j]*D[i][k];
                j++;
            }
            D[i][k]=0;
            }

        if(id==0)
        {
            for(int i=0;i<thread_num-1;i++)
                sem_wait(&sem_leader);
            for(int i=0;i<thread_num-1;i++)
                sem_post(&sem_Elimination[i]);
        }
        else
        {
            sem_post(&sem_leader);
            sem_wait(&sem_Elimination[id-1]);

        }
    }
    pthread_exit(nullptr);
}

void signal_xunhuan()
{
    long long head,tail,freq ; // timers
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq );
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    sem_init(&sem_leader,0,0);
    for(int i=0;i<thread_num-1;i++)
    {
        sem_init(&sem_Division[i],0,0);
        sem_init(&sem_Elimination[i],0,0);
    }
    pthread_t *handles=new pthread_t[thread_num];
    thread_art *param=new thread_art[thread_num];
    for(int i=0;i<thread_num;i++)
    {
        param[i].t_id=i;
        pthread_create(&handles[i],nullptr,thread_func2,(void*)&param[i]);
    }
    for(int i=0;i<thread_num;i++)
        {
            pthread_join(handles[i],nullptr);

        }

    for(int i=0;i<thread_num-1;i++)
    {
        sem_destroy(&sem_workerend[i]);
        sem_destroy(&sem_workerstart[i]);
    }
    sem_destroy(&sem_main);
    QueryPerformanceCounter((LARGE_INTEGER *)&tail );
//	cout <<"静态循环"<<(tail-head)*1000.0/freq<<"ms"<<endl;
cout <<(tail-head)*1000.0/freq<<"            ";
//    for(int i=0;i<n;i++)
//    {
//        for(int j=0;j<n;j++)
//        {
//            cout<<B[i][j]<<" ";
//            if(j==n-1)
//                cout<<endl;
//        }
//    }
}

//静态barrierE
pthread_barrier_t Divsion;
pthread_barrier_t Elimination;
void *thread_func3(void *parm)
{
    thread_art*p=(thread_art*)parm;
    int id=p->t_id;
    for(int k=0;k<n;++k)
    {
        if(id==0)
        {
            for(int j=k+1;j<n;j++)
                E[k][j]/=E[k][k];
            E[k][k]=1.0;
        }
        pthread_barrier_wait(&Divsion);
        for(int i=k+1+id;i<n;i+=thread_num)
        {
             __m128 vaik=_mm_set1_ps(E[i][k]);
            int j=k+1;
            for(j=k+1;j+4<=n;j+=4)
            {
                __m128 vakj=_mm_loadu_ps(&E[k][j]);
                __m128 vaij=_mm_loadu_ps(&E[i][j]);
                __m128 vx=_mm_mul_ps(vakj,vaik);
                vaij=_mm_sub_ps(vaij,vx);

                _mm_storeu_ps(&E[i][j], vaij);

            }
            while(j<n)
            {
                E[i][j]-=E[k][j]*E[i][k];
                j++;
            }
            E[i][k]=0;
        }
        pthread_barrier_wait(&Elimination);

    }
    pthread_exit(nullptr);
}
void pthre_barrier()
{
    long long head,tail,freq ; // timers
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq );
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    pthread_barrier_init(&Divsion, nullptr, thread_num);
    pthread_barrier_init(&Elimination, nullptr, thread_num);

    pthread_t *handles=new pthread_t[thread_num];
    thread_art *param=new thread_art[thread_num];
    for(int i=0;i<thread_num;i++)
    {
        param[i].t_id=i;
        pthread_create(&handles[i],nullptr,thread_func3,(void*)&param[i]);
    }
    for(int i=0;i<thread_num;i++)
        {
            pthread_join(handles[i],nullptr);

        }

    pthread_barrier_destroy(&Divsion);
    pthread_barrier_destroy(&Elimination);


    QueryPerformanceCounter((LARGE_INTEGER *)&tail );


//	cout <<"barrier"<<(tail-head)*1000.0/freq<<"ms"<<endl;
cout <<(tail-head)*1000.0/freq<<"            ";
//    for(int i=0;i<n;i++)
//    {
//        for(int j=0;j<n;j++)
//        {
//            cout<<D[i][j]<<" ";
//            if(j==n-1)
//                cout<<endl;
//        }
//    }
}
int main()
{
    while(n<2000)
    {
        cout<<n<<"           ";
        init(n);
        deepcopy();
        chuanxing();
        dynamic_simd();
        signal();
        signal_xunhuan();
        pthre_barrier();
        //dynamic_simd();
        cout<<endl;
        n+=100;

    }

}
