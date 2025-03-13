package main

import (
	"bytes"
	"fmt"
	"image"
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

func loadPixels(uniqueColors map[string] bool, img image.Image) {
	bounds := img.Bounds()

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			pixelColor := img.At(x, y)
			r, g, b, a := pixelColor.RGBA()
			
			// Convert to 8-bit color values
			r, g, b, a = r>>8, g>>8, b>>8, a>>8
			
			// Create a string key for the color
			colorKey := fmt.Sprintf("RGBA(%d,%d,%d,%d)", r, g, b, a)
			
			// Add to our map/"set"
			uniqueColors[colorKey] = true
		}
	}
}

func main() {
	wd, _ := os.Getwd()
	dataset_dir := path.Join(wd, "dataset")

	expected_width := 64
	expected_height := 64
	uniqueColors := make(map[string] bool)

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

			loadPixels(uniqueColors, img)
		}
	}

	log("all images have same resolution")
	for color := range uniqueColors {
		log(color)
	}
}
