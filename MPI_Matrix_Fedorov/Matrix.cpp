#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <random>
#include <string>

#include "mpi.h"

constexpr int size = 500;

	int main() {
		auto A = new float[size][size]{};
		auto B = new float[size][size]{};

		int tag = 0;
		int nProcesses, rank;

		MPI_Init(NULL, NULL);
		MPI_Status status;
		MPI_Comm comm = MPI_COMM_WORLD;
		MPI_Comm_size(comm, &nProcesses);
		MPI_Comm_rank(comm, &rank);

		if (rank == 0) {
			// start randomize
			std::random_device rd;
			std::default_random_engine eng(rd());
			std::uniform_real_distribution<> getFloat(-RAND_MAX, RAND_MAX);

			auto createFloats{
					[&A, &B, &getFloat, &eng](unsigned int from, unsigned int to) {
						for (unsigned int i = from; i < to; ++i) {
							for (unsigned int j = 0; j < size; ++j) {
								A[i][j] = (float)getFloat(eng);
								B[i][j] = (float)getFloat(eng);
							}
						}
					}
			};

			std::vector<std::thread> threadVectorRandomizer;
			unsigned int nThreads = std::thread::hardware_concurrency();
			unsigned int step = size / nThreads;

			for (unsigned int k = 0; k < nThreads; ++k) {
				unsigned int remains = (k == nThreads - 1) ? size % nThreads : 0;
				threadVectorRandomizer.emplace_back(createFloats, k * step, (k + 1) * step + remains);
			}

			for (auto& thread : threadVectorRandomizer) {
				thread.join();
			}
			// end randomize

			std::cout << "A and B filled" << std::endl;
			step = size / (nProcesses == 1 ? 1 : (nProcesses - 1));

			auto start = std::chrono::system_clock::now();

			for (int i = 1; i < nProcesses; i++) {
				int remains = (i == nProcesses - 1) ? (int)size % (nProcesses - 1) : 0;
				int from = (i - 1) * step;
				int to = i * step + remains;

				MPI_Send(&(A[0][0]),    size * size, MPI_FLOAT, i, tag,     comm); // A
				MPI_Send(&(B[0][0]),    size * size, MPI_FLOAT, i, tag + 1, comm); // B
				MPI_Send(&from,			4,           MPI_INT,   i, tag + 2, comm); // from
				MPI_Send(&to,			4,           MPI_INT,   i, tag + 3, comm); // to
			}

			auto result = new float[size][size]{};

			for (int i = 1; i < nProcesses; i++) {
				int remains = (i == nProcesses - 1) ? (int)size % (nProcesses - 1) : 0;
				int from = (i - 1) * step;
				int to = i * step + remains;

				int recvSize;
				auto tempResult = new float[to - from][size]{};
				MPI_Probe(i, tag, comm, &status);
				MPI_Recv(&(tempResult[0][0]), (to - from) * size, MPI_FLOAT, i, tag, comm, &status); // temp result
				std::cout << rank << " got from " << i << " ";
				MPI_Get_count(&status, MPI_FLOAT, &recvSize);
				std::cout << recvSize << " elements" << std::endl;

				for (int i1 = from; i1 < to; i1++) {
					for (int j = 0; j < size; j++) {
						result[i1][j] = tempResult[i1 - from][j];
					}
				}
			}

			auto end = std::chrono::system_clock::now();
			auto elapsed_nanoseconds = (end - start).count();
			std::cout << "MPI-8 time spent, ns: " << elapsed_nanoseconds << std::endl;

			start = std::chrono::system_clock::now();
			auto resultSingleThread = new float[size][size]{};

			for (size_t i2 = 0; i2 < size; ++i2) {
				for (size_t j = 0; j < size; ++j) {
					resultSingleThread[i2][j] = 0;

					for (size_t j2 = 0; j2 < size; ++j2) {
						resultSingleThread[i2][j] += A[i2][j2] * B[j2][j];
					}
				}
			}

			end = std::chrono::system_clock::now();
			elapsed_nanoseconds = (end - start).count();
			std::cout << "1 th. time spent, ns: " << elapsed_nanoseconds << std::endl;

			for (int r1 = 0; r1 < size; ++r1) {
				for (int r2 = 0; r2 < size; ++r2) {
					if (result[r1][r2] != resultSingleThread[r1][r2]) {
						std::cerr << 
							result[r1][r2] << " not equals " << resultSingleThread[r1][r2] << " result[" << r1 << "][" << r2 << "]"
								<< std::endl;
						r1 = size - 1;
						break;
					}
				}
			}
		} else {
			int flag = 0;
			int from, to;

			MPI_Iprobe(0, tag,     comm, &flag, &status);
			MPI_Iprobe(0, tag + 1, comm, &flag, &status);
			MPI_Iprobe(0, tag + 2, comm, &flag, &status);
			MPI_Iprobe(0, tag + 3, comm, &flag, &status);
			std::cout << rank << "   probe   success" << std::endl;
			MPI_Recv(&(A[0][0]),    size * size, MPI_FLOAT, 0, tag,     comm, &status); // A
			MPI_Recv(&(B[0][0]),    size * size, MPI_FLOAT, 0, tag + 1, comm, &status); // B
			MPI_Recv(&from,			4,           MPI_INT,   0, tag + 2, comm, &status); // from
			MPI_Recv(&to,			4,           MPI_INT,   0, tag + 3, comm, &status); // to
			std::cout << rank << " recv success: " << from << "-" << to << std::endl;

			auto tempResult = new float[to - from][size]{};

			for (int i = from; i < to; ++i) {
				for (int j = 0; j < size; ++j) {
					tempResult[i - from][j] = 0;

					for (int j2 = 0; j2 < size; ++j2) {
						tempResult[i - from][j] += A[i][j2] * B[j2][j];
					}
				}
			}

			MPI_Send(&(tempResult[0][0]), (to - from) * size, MPI_FLOAT, 0, tag, comm); // result
			std::cout << rank << " computed" << std::endl;
		}

		std::cout << rank << " finalizing" << std::endl;
		MPI_Finalize();
	}