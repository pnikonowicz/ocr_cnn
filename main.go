package main

import (
	"fmt"
	"os"
	"path"
)

func log(message string) {
	fmt.Println(message)
}

func main() {
	wd, _ := os.Getwd()
	dataset_dir := path.Join(wd, "dataset")

	for number := range []int{0,1,2,3,4,5,6,7,8,9} {

		dir := path.Join(dataset_dir, string(rune('0' + number)))
		log(dir)
		dir_iterator, err := os.ReadDir(dir)

		if(err != nil) {
			log("could not read dir")
			os.Exit(1)
		}

		for _, file := range dir_iterator {
			fmt.Println(file, file.IsDir())
		}
	}

}
