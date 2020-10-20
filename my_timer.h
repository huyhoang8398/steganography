#pragma once
#include <chrono>

class MyTimer
{
private:
	std::chrono::time_point<std::chrono::steady_clock> starter;
	std::chrono::time_point<std::chrono::steady_clock> ender;

public:
	void startCounter() {
		starter = std::chrono::steady_clock::now();
	}

	double getCounter() {
		ender = std::chrono::steady_clock::now();
		return double(std::chrono::duration_cast<std::chrono::nanoseconds>(ender - starter).count()) / 1000000; // millisecond output
	}
};
