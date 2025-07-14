source "https://rubygems.org"

ruby "3.2.0"

# Jekyll
gem "jekyll", "~> 4.3"

# Jekyll plugins
group :jekyll_plugins do
  gem "jekyll-feed", "~> 0.12"
  gem "jekyll-sitemap"
  gem "jekyll-seo-tag"
  gem "jekyll-paginate", "~> 1.1"
  gem "jekyll-gist"
  gem "jekyll-include-cache"
end

# Windows and JRuby does not include zoneinfo files, so bundle the tzinfo-data gem
# and associated library.
platforms :mingw, :x64_mingw, :mswin, :jruby do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
end

# Performance-booster for watching directories on Windows
gem "wdm", "~> 0.1.1", :platforms => [:mingw, :x64_mingw, :mswin]

# Lock `http_parser.rb` gem to `v0.6.x` on JRuby builds since newer versions of the gem
# do not have a Java counterpart.
gem "http_parser.rb", "~> 0.6.0", :platforms => [:jruby]

# Webrick for Ruby 3.0+
gem "webrick", "~> 1.8"

# Rouge for syntax highlighting
gem "rouge", "~> 3.30"

# Kramdown for markdown processing
gem "kramdown", "~> 2.4"
gem "kramdown-parser-gfm", "~> 1.1"

# For development
group :development do
  # gem "jekyll-admin"
end