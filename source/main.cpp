#include "FaceSwapper.hpp"


int main(int argc, char **argv)
{
	FaceSwapper *app = new FaceSwapper();

	app->init();

	while(app->running())
	{
		app->processInput();
		app->update();
		app->draw();
	}

	delete app;

	return 0;
}
