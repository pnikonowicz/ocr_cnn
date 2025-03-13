package main

import (
	"bytes"
	"fmt"
	"image/png"
	"os"
	"path"
)

func log(message string) {
	fmt.Println(message)
}

func printAndTerminate(message string) {
	log(message)
	os.Exit(1)
}

func main() {
	wd, _ := os.Getwd()
	dataset_dir := path.Join(wd, "dataset")

	expected_width := 64
	expected_height := 64

	for number := range []int{0,1,2,3,4,5,6,7,8,9} {

		dir := path.Join(dataset_dir, string(rune('0' + number)))
		dir_iterator, err := os.ReadDir(dir)

		log(fmt.Sprintf("checking dir: %s", dir))

		if(err != nil) {
			printAndTerminate("could not read dir")
		}

		for _, file_entry := range dir_iterator {
			file_name := path.Join(dir, file_entry.Name())
			file_contents, err := os.ReadFile(file_name)
			if err != nil {
				printAndTerminate(fmt.Sprintf("could not read file: %s", file_name))
			}

			img, err := png.Decode(bytes.NewReader(file_contents))
			if err != nil {
				printAndTerminate(fmt.Sprintf("could not read png: %s", file_name))
			}

			bounds := img.Bounds()
			width := bounds.Max.X - bounds.Min.X
			height := bounds.Max.Y - bounds.Min.Y

			if width != expected_width || height != expected_height {
				printAndTerminate(fmt.Sprintf("bad image: w: %d h: %d", width, height))
			}
		}
	}

	log("all images have same resolution")
}
