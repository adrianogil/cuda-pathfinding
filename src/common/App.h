#ifndef _APP_H_
#define _APP_H_

#ifdef _WIN32
   #include <windows.h>
   /**
     * \brief Estrutura para tratar cron�metro.
     */
   struct stStopwatch_Win
   {
      LARGE_INTEGER mStartTime;
      LARGE_INTEGER mEndTime;
      double mCPUFreq;
      double mElapsedTime;
   };
   typedef struct stStopwatch_Win Stopwatch;

   #define USER_PAUSE system("PAUSE>null");

   /**
     * \brief Inicializa cron�metro.
     */
   #define FREQUENCY( prm ) {                                                    \
      LARGE_INTEGER CPU_freq;                                                    \
      double mCPUFreq  ;                                                         \
      QueryPerformanceFrequency((LARGE_INTEGER*) &CPU_freq);                     \
      mCPUFreq = static_cast <double> (CPU_freq.QuadPart);                       \
      mCPUFreq /= 1000.0f;                                                       \
      QueryPerformanceCounter((LARGE_INTEGER*) &prm.mStartTime);                 \
      QueryPerformanceCounter((LARGE_INTEGER*) &prm.mEndTime);                   \
      prm.mElapsedTime = 0.0f;                                                   \
      prm.mCPUFreq = mCPUFreq;                                                   \
   }


   /**
     * \brief Dispara cron�metro.
     */
   #define START_STOPWATCH( prm ) {                                               \
      QueryPerformanceCounter((LARGE_INTEGER*) &prm.mStartTime);                 \
   }

   /**
     * \brief Para cron�metro.
     */
   #define STOP_STOPWATCH( prm ) {                                                     \
      QueryPerformanceCounter((LARGE_INTEGER*) &prm.mEndTime);                         \
      prm.mElapsedTime = (double) (prm.mEndTime.QuadPart -  prm.mStartTime.QuadPart);  \
      prm.mElapsedTime /= prm.mCPUFreq;                                                \
   }

#define ELAPSED_TIME(stop_watch, calls) { \
	stStopwatch_Win stop_watch; \
   FREQUENCY(stop_watch); \
   START_STOPWATCH(stop_watch); \
   calls \
   STOP_STOPWATCH(stop_watch); \
   printf("\nCurrent time: %f\n", stop_watch.mElapsedTime); \
   }\

#define ELAPSED_TIME(stop_watch, text, calls) { \
	stStopwatch_Win stop_watch; \
   FREQUENCY(stop_watch); \
   START_STOPWATCH(stop_watch); \
   calls \
   STOP_STOPWATCH(stop_watch); \
   printf(text, stop_watch.mElapsedTime); \
      }\


#define ELAPSED_TIME(stop_watch, text, calls) { \
	stStopwatch_Win stop_watch; \
   FREQUENCY(stop_watch); \
   START_STOPWATCH(stop_watch); \
   calls \
   STOP_STOPWATCH(stop_watch); \
   printf(text, stop_watch.mElapsedTime); \
         }\

#define CUDA_ELAPSED_TIME(text, calls) {\
	cudaEvent_t beginEvent, endEvent;\
   cudaEventCreate(&beginEvent); \
   cudaEventCreate(&endEvent);\
   cudaEventRecord(beginEvent, 0);\
   calls \
   cudaEventRecord(endEvent, 0);\
   cudaEventSynchronize(endEvent);\
   CUDA_CHECK_RETURN(cudaDeviceSynchronize());\
   float gpu_processing;\
   cudaEventElapsedTime(&gpu_processing, beginEvent, endEvent);\
   cudaEventDestroy(beginEvent);\
   cudaEventDestroy(endEvent);\
   printf(text, gpu_processing);\
               }\


#else
   #include <time.h>
   #include <sys/time.h>
   /**
     * \brief Estrutura para tratar cron�metro.
     */
   struct stStopwatch_Unix
   {
      struct timeval mStartTime;
      struct timeval mEndTime;
      double mCPUFreq;
      double mElapsedTime;
   };
   typedef struct stStopwatch_Unix Stopwatch;

   /**
     * \brief Inicializa cron�metro.
     */
   #define FREQUENCY( prm ) {                                                    \
      prm.mCPUFreq = 0.0f;                                                       \
      prm.mElapsedTime = 0.0001f;                                                \
   }


   /**
     * \brief Dispara cron�metro.
     */
   #define START_STOPWATCH( prm ) {                                               \
       gettimeofday( &prm.mStartTime, 0);                                   \
   }

   /**
     * \brief Para cron�metro.
     */
   #define STOP_STOPWATCH( prm ) {                                                     \
      gettimeofday( &prm.mEndTime, 0);                                                     \
      prm.mElapsedTime = (1000.0f * ( prm.mEndTime.tv_sec - prm.mStartTime.tv_sec) + (0.001f * (prm.mEndTime.tv_usec - prm.mStartTime.tv_usec)) );                                                \
   }

#endif

#endif
