APP_NAME = fusion

SRC_DIR = src
OBJS = $(SRC_DIR)/layer.o $(SRC_DIR)/model.o $(SRC_DIR)/main.o

CXX = g++
CXXFLAGS = -Wall -O3 -std=c++17 -m64 -I$(SRC_DIR)

all: $(APP_NAME)

$(APP_NAME): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp $(SRC_DIR)/%.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	/bin/rm -rf *~ $(SRC_DIR)/*.o $(APP_NAME) *.class