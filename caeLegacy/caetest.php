<?php
$root = '';
$path = 'images/';

$imgList = getImagesFromDir($root . $path);
$img = getRandomFromArray($imgList);

function getImagesFromDir($path) {
    $images = array();
    if ( $img_dir = @opendir($path) ) {
        while ( false !== ($img_file = readdir($img_dir)) ) {
            // checks for gif, jpg, png
            if ( preg_match("/(\.gif|\.jpg|\.png)$/", $img_file) ) {
                $images[] = $img_file;
            }
        }
        closedir($img_dir);
    }
    return $images;
}

function getRandomFromArray($ar) {
    $num = array_rand($ar);
    return $ar[$num];
}
?>

<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Demo</title>
<style type="text/css">
</style>
</head>

<!-- image displays here -->
<div><img src="<?php echo $path . $img ?>" alt="" /></div>

</body>
</html>
