TestConv: test_main.cpp Mat/allocator.cpp Mat/allocator.cpp x86/conv_x86.cpp
	g++ -fopenmp test_main.cpp Mat/allocator.cpp Mat/mat.cpp x86/conv_x86.cpp -I./Mat -o TestConv