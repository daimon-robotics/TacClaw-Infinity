cmake_minimum_required(VERSION 2.8.3)
cmake_policy(SET CMP0048 NEW)

project(dm25_pr02_hr_pd04_eg_20260810 VERSION 2.0.0)

find_package(catkin REQUIRED)

catkin_package()

find_package(roslaunch)

foreach(dir config meshes urdf)
	install(DIRECTORY ${dir}/
		DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/${dir})
endforeach(dir)

install(FILES README.md
	DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
